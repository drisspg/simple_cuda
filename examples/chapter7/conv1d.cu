#include "src/include/utils.h"
#include "src/include/tensors.h"

#include <cstddef>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using namespace simple_cuda;
using KernelFunc = void (*)(const float *, const float *, float *, int);

using one_d  = Extent<1>;

template <int tile_size, int filter_radius>
__global__ void Conv1D(const float *input, const float *filter, float *output,
                       const int numel) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= numel) {
    // Early return for invalid output elements
    return;
  }

  float accumulator = 0.0;
#pragma unroll
  for (int j{-filter_radius}; j <= filter_radius; j++) {
    if (0 < global_idx + j < numel) {
      accumulator += input[global_idx + j] * filter[j + filter_radius];
    }
  }
  // We early returned so we can write freely
  output[global_idx] += accumulator;
}

template <int tile_size, int filter_radius>
__device__ void fill_tile(const float *input, float *input_tile,
                          const int global_idx, const int input_size) {
  // Load the non ghost cells
  input_tile[threadIdx.x + filter_radius] = global_idx < input_size? input[global_idx]: 0.0;

  // Edges load in ghost cells
  if (static_cast<int>(threadIdx.x - filter_radius) < 0) {
    const int offset = global_idx - filter_radius;
    input_tile[threadIdx.x] = offset >= 0 ? input[offset] : 0.0;
  }
  if (threadIdx.x + filter_radius >= tile_size) {
    const int offset = global_idx + filter_radius;
    input_tile[filter_radius + threadIdx.x + filter_radius] =
        offset < input_size ? input[offset] : 0.0;
  }
}

template <int tile_size, int filter_radius>
__global__ void Conv1D_shmem(const float *input, const float *filter,
                             float *output, const int input_size) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // We stored the the tile_size which is equivelant to block size + the filter
  // size for the left and right boundaries
  __shared__ float input_tile[tile_size + (2 * filter_radius)];

  fill_tile<tile_size, filter_radius>(input, input_tile, global_idx, input_size);
  __syncthreads();
  float accumulator = 0.0;
#pragma unroll
  for (int j{-filter_radius}; j <= filter_radius; j++) {
    const int tile_idx = filter_radius + threadIdx.x + j;
    accumulator += input_tile[tile_idx] * filter[j + filter_radius];
  }

  if (global_idx < input_size) {
    output[global_idx] = accumulator;
  }
}

template <typename T> T cpp_kernel(T const &input, T const &filter) {
  const auto n_elements = input.size();
  const auto kernel_size = filter.size();
  std::vector<float> output;
  output.reserve(input.size());
  const int kernel_width = kernel_size / 2;
  for (int i{0}; i < n_elements; i++) {
    float accum = 0.0;
    for (int j{-kernel_width}; j <= kernel_width; j++) {
      if (0 <= i + j && i + j < n_elements) {
        accum += input[i + j] * filter[j + kernel_width];
      }
    }
    output.emplace_back(accum);
  }
  return output;
}

void Test(KernelFunc func, const size_t n_rows, const size_t n_cols, const size_t filter_radius,
          dim3 grid, dim3 block) {
  one_d tensor_extents({n_rows});
  one_d filter_extents({2*filter_radius});

  HostTensor<float, one_d> input_vec(tensor_extents);
  HostTensor<float, one_d> output_vec(tensor_extents);
  HostTensor<float, one_d> filter(filter_extents);

  std::fill(input_vec.data_.begin(), input_vec.data_.end(), 1);
  std::fill(output_vec.data_.begin(), output_vec.data_.end(), 0);
  std::fill(filter.data_.begin(), filter.data_.end(), 2);

  auto input_vec_d = input_vec.to_device();
  auto output_vec_d = output_vec.to_device();
  auto filter_d = filter.to_device();

  func<<<grid, block>>>(input_vec_d.data_ptr(), filter_d.data_ptr(), output_vec_d.data_ptr(), tensor_extents.numel());
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  auto host_output = output_vec_d.to_host();
  auto host_output_ptr = host_output.data_ptr();

  const auto cpp_anwser = cpp_kernel(input_vec.data_, filter.data_);

  for (const int idx : std::views::iota(0, int(tensor_extents.numel()))) {
    if (host_output_ptr[idx] != cpp_anwser[idx]) {
      std::string error_string = "Houston we have a problem!\n";
      error_string += fmt::format("At ({}) found value: {} instead of {}!\n",
                                  idx, host_output_ptr[idx], cpp_anwser[idx]);
      std::cout << error_string;

      if (tensor_extents.numel() <= 32) {
        fmt::print("Good:{}\n", fmt::join(cpp_anwser, ", "));
        fmt::print("Bad:{}\n", fmt::join(host_output.data_, ", "));
      }

      exit(1);
    }
  }
  std::cout << "All good brother!\n";
}

int main() {
  // Standard Matmul
  constexpr int max_length = 10000;
  constexpr int filter_radius = 21;
  constexpr int block_size = 1024;

  // dimx is inner dim, dimy is outerdim
  dim3 grid(ceil_div(max_length, block_size));
  dim3 block(block_size);

  Test(Conv1D<block_size, filter_radius>, max_length, max_length, filter_radius, grid,
       block);

  Test(Conv1D_shmem<block_size, filter_radius>, max_length, max_length, filter_radius, grid,
       block);

  // profile the relevant kernels:
  // ncu -k "regex:Conv" ./bin/conv1d
  return 0;
}