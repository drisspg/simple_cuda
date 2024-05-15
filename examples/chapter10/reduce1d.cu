#include "src/include/tensors.h"
#include "src/include/utils.h"

#include <cstddef>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <numeric>
#include <cmath>
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

float cpp_kernel(std::vector<float> &input) {
  const auto n_elements = input.size();
  std::vector<float> input_copy(input.size());
  std::copy(input.begin(), input.end(), input_copy.begin());
  auto out = std::reduce(input_copy.begin(), input_copy.end());
  return out;
}

void Test(KernelFunc func, const size_t numel, dim3 grid, dim3 block) {
  one_d tensor_extents({numel});

  HostTensor<float, one_d> input_vec(tensor_extents);
  HostTensor<float, one_d> out_sum(one_d({1}));

  fill_random(input_vec.data_);
  std::fill(out_sum.data_.begin(), out_sum.data_.end(), 0);

  auto input_vec_d = input_vec.to_device();
  auto out_sum_d = out_sum.to_device();

  func<<<grid, block>>>(input_vec_d.data_ptr(), out_sum_d.data_ptr(),
                        tensor_extents.numel());
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  auto host_output = out_sum_d.to_host();
  auto host_output_ptr = host_output.data_ptr();

  std::vector<float> input_vector(input_vec.data_.begin(),
                                  input_vec.data_.end());
  const auto cpp_anwser = cpp_kernel(input_vector);

  float diff = fabs(cpp_anwser - host_output_ptr[0]);
  if (diff > 1e-3) {
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
  // Standard Matmul
  constexpr int max_length = 1024;
  constexpr int block_size = max_length/2;

  dim3 grid(1);
  dim3 block(block_size);

  Test(Reduce1dInplace, max_length, grid, block);

  // profile the relevant kernels:
  // ncu -k "regex:reduce" ./bin/conv1d
  return 0;
}
