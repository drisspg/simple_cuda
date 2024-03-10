#include "src/include/tensors.h"
#include "src/include/utils.h"

#include <cstddef>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using namespace simple_cuda;
using KernelFunc = void (*)(const float *, const float *, float *, const int,
                            const int);

using two_d = Extent<2>;

template <int tile_size, int filter_radius>
__global__ void Conv2D(const float *input, const float *filter, float *output,
                       const int n_rows, const int n_cols) {
  const int global_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int filter_row_stride = 2 * filter_radius;
  const int row_stride = n_cols;
  const int col_stride = 1;
  const int global_idx =
      global_row_idx * row_stride + global_col_idx * col_stride;
  if (global_row_idx >= n_rows || global_col_idx >= n_cols) {
    // Early return for invalid output elements
    return;
  }

  float accumulator = 0.0;
#pragma unroll
  for (int k{-filter_radius}; k < filter_radius; k++) {
    for (int l{-filter_radius}; l < filter_radius; l++) {
      const int effective_row = global_row_idx + k;
      const int effective_col = global_col_idx + l;
      if (0 <= effective_row && effective_row < n_rows && 0 <= effective_col &&
          effective_col < n_cols) {
        accumulator +=
            input[effective_row * row_stride + effective_col * col_stride] *
            filter[(k + filter_radius) * filter_row_stride +
                   (l + filter_radius) * 1];
      }
    }
  }
  // We early returned so we can write freely
  output[global_idx] += accumulator;
}

template <int tile_size, int filter_radius>
__device__ void fill_tile(const float *input, float *input_tile,
                          const int global_idx, const int input_size) {
  // Load the non ghost cells
  input_tile[threadIdx.x + filter_radius] =
      global_idx < input_size ? input[global_idx] : 0.0;

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

template <typename T>
T cpp_kernel(T const &input, T const &filter, const int n_rows,
             const int n_cols, const int filter_radius) {
  std::vector<float> output;
  output.reserve(input.size());
  const int filter_row_stride = 2 * filter_radius;
  const int row_stride = n_cols;
  const int col_stride = 1;
  for (int i{0}; i < n_rows; i++) {
    for (int j{0}; j < n_cols; j++) {
      float accum = 0.0;
      for (int k{-filter_radius}; k < filter_radius; k++) {
        for (int l{-filter_radius}; l < filter_radius; l++) {
          const int effective_row = i + k;
          const int effective_col = j + l;
          if (0 <= effective_row && effective_row < n_rows &&
              0 <= effective_col && effective_col < n_cols) {
            accum +=
                input[effective_row * row_stride + effective_col * col_stride] *
                filter[(k + filter_radius) * filter_row_stride +
                       (l + filter_radius)];
          }
        }
      }
      output.emplace_back(accum);
    }
  }

  return output;
}

void Test(KernelFunc func, const size_t n_rows, const size_t n_cols,
          const size_t filter_radius, dim3 grid, dim3 block) {
  two_d tensor_extents({n_rows, n_cols});
  two_d filter_extents({(2 * filter_radius), (2 * filter_radius)});

  HostTensor<float, two_d> input_vec(tensor_extents);
  HostTensor<float, two_d> output_vec(tensor_extents);
  HostTensor<float, two_d> filter(filter_extents);

  std::fill(input_vec.data_.begin(), input_vec.data_.end(), 1);
  std::fill(output_vec.data_.begin(), output_vec.data_.end(), 0);
  std::fill(filter.data_.begin(), filter.data_.end(), 2);

  auto input_vec_d = input_vec.to_device();
  auto output_vec_d = output_vec.to_device();
  auto filter_d = filter.to_device();
  cudaCheckErrors("Thrust host to device failed!");

  func<<<grid, block>>>(input_vec_d.data_ptr(), filter_d.data_ptr(),
                        output_vec_d.data_ptr(), tensor_extents.size<0>(),
                        tensor_extents.size<1>());
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  auto host_output = output_vec_d.to_host();
  auto host_output_ptr = host_output.data_ptr();

  const auto cpp_anwser =
      cpp_kernel(input_vec.data_, filter.data_, n_rows, n_cols, filter_radius);

  for (const int idx : std::views::iota(0, int(tensor_extents.numel()))) {
    if (host_output_ptr[idx] != cpp_anwser[idx]) {
      std::string error_string = "Houston we have a problem!\n";
      error_string += fmt::format("At ({}) found value: {} instead of {}!\n",
                                  idx, host_output_ptr[idx], cpp_anwser[idx]);
      std::cout << error_string;

      if (tensor_extents.numel() <= 100) {
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
  constexpr int num_rows = 256;
  constexpr int num_cols = 256;
  constexpr int filter_radius = 4;
  constexpr int block_size = 32;

  // dimx is inner dim, dimy is outerdim
  dim3 grid(ceil_div(num_rows, block_size), ceil_div(num_cols, block_size));
  dim3 block(block_size, block_size);

  Test(Conv2D<block_size, filter_radius>, num_rows, num_cols, filter_radius,
       grid, block);

  return 0;
}