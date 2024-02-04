#include "src/include/utils.h"
#include <array>
#include <fmt/core.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

using namespace simple_cuda;

__global__ void saturated_cast(nv_bfloat16 *input, __nv_fp8_storage_t *output,
                               int n_rows, int n_cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Assume row major
  const int global_index = row * n_cols + col;
  if (row < n_rows && col < n_cols) {
    output[global_index] = __nv_cvt_bfloat16raw_to_fp8(
        input[global_index], __nv_saturation_t::__NV_SATFINITE,
        __nv_fp8_interpretation_t::__NV_E4M3);
  }
}

void test_input_size(int n_rows, int n_cols, int block_size) {
  thrust::device_vector<nv_bfloat16> input(n_rows * n_cols);
  thrust::fill(input.begin(), input.end(), 448.0f);
  thrust::device_vector<__nv_fp8_storage_t> output(n_rows * n_cols);

  dim3 grid(ceil_div(n_cols, block_size), ceil_div(n_rows, block_size));
  dim3 block(block_size, block_size);
  auto kerneltime = kernel_time([&]() {
    saturated_cast<<<grid, block>>>(input.data().get(), output.data().get(),
                                    n_rows, n_cols);
  });
  fmt::print("Kernel time: {} in usecond\n", kerneltime * 1000);

  // print first element of ouutput
  thrust::host_vector<__nv_fp8_storage_t> output_host = output;
  fmt::print("The first elemnt is: {0:b}\n", output_host[0]);
}

int main() {
  // Standard Matmul
  auto n_rows = std::to_array({256, 512, 4096});
  auto n_cols = std::to_array({256, 512, 4096});

  for (auto row : n_rows) {
    for (auto col : n_cols) {
      fmt::print("Testing input size: {}x{}\n", row, col);
      test_input_size(row, col, 32);
    }
  }

  // NCU reports a factor of 10 less time?!

  return 0;
}