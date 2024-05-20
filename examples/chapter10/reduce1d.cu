#include "src/include/tensors.h"
#include "src/include/utils.h"

#include <cstddef>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <cmath>
#include <numeric>
#include <optional>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using namespace simple_cuda;
using KernelFunc = void (*)(float *, float *, const int);

using one_d = Extent<1>;

__global__ void Reduce1dInplace(float *input, float *output, const int numel) {
  const int i = 2 * threadIdx.x;
  for (unsigned stride{1}; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

__global__ void Reduce1dInplaceBetterOrdering(float *input, float *output,
                                              const int numel) {
  const int i = threadIdx.x;
  for (unsigned stride{blockDim.x}; stride >= 1; stride /= 2) {
    if (i < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *output = input[0];
  }
}

__global__ void Reduce1dShared(float *input, float *output, const int numel) {
  const int i = threadIdx.x;
  extern __shared__ float shmem[];
  // First iter pulled out of loop
  shmem[i] = input[i] + input[i + blockDim.x];
  __syncthreads();
  for (unsigned stride{blockDim.x / 2}; stride >= 1; stride /= 2) {
    if (i < stride) {
      shmem[i] += shmem[i + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *output = shmem[0];
  }
}

__global__ void Reduce1dSharedGlobal(float *input, float *output,
                                     const int numel) {
  const int local_id = threadIdx.x;
  const int global_id = local_id + 2 * blockDim.x * blockIdx.x;
  extern __shared__ float shmem[]; // Size blockDim.x
  // First iter pulled out of loop
  shmem[local_id] = input[global_id] + input[global_id + blockDim.x];
  __syncthreads();
  for (unsigned stride{blockDim.x / 2}; stride >= 1; stride /= 2) {
    if (local_id < stride) {
      shmem[local_id] += shmem[local_id + stride];
    }
    __syncthreads();
  }

  if (local_id == 0) {
    atomicAdd(output, shmem[0]);
  }
}

template <int COARSE_FACTOR>
__global__ void Reduce1dSharedGlobalCoarse(float *input, float *output,
                                           const int numel) {
  const int local_id = threadIdx.x;
  const int global_offset = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
  const int global_id = local_id + global_offset;
  extern __shared__ float shmem[]; // Size blockDim.x
  // First iter pulled out of loop
  float sum = input[global_id];
#pragma unroll
  for (int tile = 1; tile < COARSE_FACTOR * 2; tile++) {
    sum += input[global_id + tile * blockIdx.x];
  }
  shmem[local_id] = sum;
  __syncthreads();
  for (unsigned stride{blockDim.x / 2}; stride >= 1; stride /= 2) {
    if (local_id < stride) {
      shmem[local_id] += shmem[local_id + stride];
    }
    __syncthreads();
  }

  if (local_id == 0) {
    atomicAdd(output, shmem[0]);
  }
}

float cpp_kernel(std::vector<float> &input) {
  const auto n_elements = input.size();
  std::vector<float> input_copy(input.size());
  std::copy(input.begin(), input.end(), input_copy.begin());
  auto out = std::reduce(input_copy.begin(), input_copy.end());
  return out;
}

void Test(KernelFunc func, const size_t numel, dim3 grid, dim3 block,
          std::optional<size_t> shmem) {
  one_d tensor_extents({numel});

  HostTensor<float, one_d> input_vec(tensor_extents);
  HostTensor<float, one_d> out_sum(one_d({1}));

  fill_random(input_vec.data_);
  // std::fill(input_vec.data_.begin(), input_vec.data_.end(), 1);
  std::fill(out_sum.data_.begin(), out_sum.data_.end(), 0);

  auto input_vec_d = input_vec.to_device();
  auto out_sum_d = out_sum.to_device();

  if (shmem.has_value()) {
    func<<<grid, block, shmem.value()>>>(
        input_vec_d.data_ptr(), out_sum_d.data_ptr(), tensor_extents.numel());

  } else {
    func<<<grid, block>>>(input_vec_d.data_ptr(), out_sum_d.data_ptr(),
                          tensor_extents.numel());
  }
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  auto host_output = out_sum_d.to_host();
  auto host_output_ptr = host_output.data_ptr();

  std::vector<float> input_vector(input_vec.data_.begin(),
                                  input_vec.data_.end());
  const auto cpp_anwser = cpp_kernel(input_vector);

  float diff = fabs(cpp_anwser - host_output_ptr[0]);
  if (diff > 5e-3) {
    std::string error_string = "Houston we have a problem!\n";
    error_string += fmt::format("Found a deviation of {}\n", diff);
    error_string += fmt::format("Cpp anwser: {}, GPU anwser: {}\n", cpp_anwser,
                                host_output_ptr[0]);
    std::cout << error_string;
    exit(1);
  }
  std::cout << "All good brother!\n";
}

int main() {
  constexpr int max_length = 2048;
  constexpr int block_size = max_length / 2;

  dim3 grid(1);
  dim3 block(block_size);

  // Base case bad ordering inplace writes
  fmt::print("• Reduced1dInplace Test: ");
  Test(Reduce1dInplace, max_length, grid, block, std::nullopt);

  // Inplace writes bad ordering
  fmt::print("• Reduced1dInplaceBetterOrdering Test: ");
  Test(Reduce1dInplaceBetterOrdering, max_length, grid, block, std::nullopt);

  // Dynamic shmem version
  fmt::print("• Reduce1dShared Test: ");
  size_t shmem{block.x * sizeof(float)};
  Test(Reduce1dShared, max_length, grid, block, shmem);

  // Test larger than thread reductions
  constexpr int max_length_global = 2048 * 2;

  block.x = 1024;
  grid.x = ceil_div(max_length_global, block.x * 2);
  shmem = block.x * sizeof(float);
  fmt::print("• Reduce1dSharedGlobal Test: ");
  Test(Reduce1dSharedGlobal, max_length_global, grid, block, shmem);

  constexpr int coarse_factor = 2;
  grid.x = ceil_div(max_length_global, block.x * 2 * coarse_factor);
  shmem = block.x * sizeof(float);
  fmt::print("• Reduce1dSharedGlobalCoarse Test: ");
  Test(Reduce1dSharedGlobalCoarse<coarse_factor>, max_length_global, grid,
       block, shmem);

  // profile the relevant kernels:
  // ncu -k "regex:reduce" ./bin/conv1d
  return 0;
}
