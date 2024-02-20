#include "src/include/utils.h"

#include <cmath>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using namespace simple_cuda;
using KernelFunc = void (*)(const float *, const float *, float *, int);

using host_vec = thrust::host_vector<float>;
using device_vec = thrust::device_vector<float>;

template <int tile_size, int filter_size>
__global__ void Conv1D(const float *input, const float *filter, float *output,
                       const int input_size) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= input_size) {
    // Early return for invalid output elements
    return;
  }
  const int kernel_width = filter_size / 2;
  float accumulator = 0.0;
#pragma unroll
  for (int j{-kernel_width}; j <= kernel_width; j++) {
    if (0 < global_idx + j < input_size) {
      accumulator += input[global_idx + j] * filter[j + kernel_width];
    }
  }
  // We early returned so we can write freely
  output[global_idx] += accumulator;
}

template <int tile_size, int filter_size>
__device__ void fill_tile(const float *input, float *input_tile,
                          const int global_idx, const int input_size,
                          const int kernel_width) {
  // Load the non ghost cells
  input_tile[threadIdx.x + kernel_width] = global_idx < input_size? input[global_idx]: 0.0;

  // Edges load in ghost cells
  if (static_cast<int>(threadIdx.x - kernel_width) < 0) {
    const int offset = global_idx - kernel_width;
    input_tile[threadIdx.x] = offset >= 0 ? input[offset] : 0.0;
  }
  if (threadIdx.x + kernel_width >= tile_size) {
    const int offset = global_idx + kernel_width;
    input_tile[kernel_width + threadIdx.x + kernel_width] =
        offset < input_size ? input[offset] : 0.0;
  }
}

template <int tile_size, int filter_size>
__global__ void Conv1D_shmem(const float *input, const float *filter,
                             float *output, const int input_size) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int kernel_width = filter_size / 2;
  // We stored the the tile_size which is equivelant to block size + the filter
  // size for the left and right boundaries
  __shared__ float input_tile[tile_size + filter_size];

  fill_tile<tile_size, filter_size>(input, input_tile, global_idx, input_size,
                                    kernel_width);
  __syncthreads();
  float accumulator = 0.0;
#pragma unroll
  for (int j{-kernel_width}; j <= kernel_width; j++) {
    const int tile_idx = kernel_width + threadIdx.x + j;
    accumulator += input_tile[tile_idx] * filter[j + kernel_width];
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

void Test(KernelFunc func, const int vec_length, const int filter_size,
          dim3 grid, dim3 block) {
  host_vec input_vec(vec_length);
  host_vec output_vec(vec_length);
  host_vec filter(filter_size);

  // fill_random(input_vec);
  // fill_random(filter);

  std::fill(input_vec.begin(), input_vec.end(), 1);
  std::fill(output_vec.begin(), output_vec.end(), 0);
  std::fill(filter.begin(), filter.end(), 2);

  device_vec input_vec_d(input_vec);
  device_vec output_vec_d(output_vec);
  device_vec filter_d(filter);

  float *input_vec_ptr = thrust::raw_pointer_cast(input_vec_d.data());
  float *output_vec_ptr = thrust::raw_pointer_cast(output_vec_d.data());
  float *filter_ptr = thrust::raw_pointer_cast(filter_d.data());

  func<<<grid, block>>>(input_vec_ptr, filter_ptr, output_vec_ptr, vec_length);
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  auto host_output = host_vec(output_vec_d);
  float *host_output_ptr = thrust::raw_pointer_cast(host_output.data());

  const auto cpp_anwser = cpp_kernel(host_vec(input_vec), host_vec(filter));

  for (const int idx : std::views::iota(0, vec_length)) {
    if (host_output_ptr[idx] != cpp_anwser[idx]) {
      std::string error_string = "Houston we have a problem!\n";
      error_string += fmt::format("At ({}) found value: {} instead of {}!\n",
                                  idx, host_output_ptr[idx], cpp_anwser[idx]);
      std::cout << error_string;

      if (vec_length < 32) {
        fmt::print("Good:{}\n", fmt::join(cpp_anwser, ", "));
        fmt::print("Bad:{}\n", fmt::join(host_output, ", "));
      }

      exit(1);
    }
  }
  std::cout << "All good brother!\n";
}

int main() {
  // Standard Matmul
  constexpr int max_length = 1000000;
  constexpr int filter_length = 21;
  constexpr int block_size = 1024;

  // dimx is inner dim, dimy is outerdim
  dim3 grid(ceil_div(max_length, block_size));
  dim3 block(block_size);

  Test(Conv1D<block_size, filter_length>, max_length, filter_length, grid,
       block);

  Test(Conv1D_shmem<block_size, filter_length>, max_length, filter_length, grid,
       block);

  // profile the relevant kernels:
  // ncu -k "regex:Conv" ./bin/conv1d
  return 0;
}