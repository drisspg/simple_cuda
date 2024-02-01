#include <fmt/core.h>
#include "src/include/utils.h"
#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using KernelFunc = void (*)(float *, float *, float *, int);

__global__ void MatrixVectorKernel(float *Matrix, float *Vector, float *Output,
                                   int vector_height) {
  // Two sqaure matrices and performs matmul
  constexpr auto matrix_col_stride = 1;
  const auto matrix_row_stride = vector_height;
  constexpr auto vector_row_stride = 1;
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (row < vector_height) {
    float accum = 0;
    for (int k = 0; k < vector_height; k++) {
      int m_idx = row * matrix_row_stride + k * matrix_col_stride;
      int n_idx = k * vector_row_stride;
      accum += Matrix[m_idx] * Vector[n_idx];
    }
    Output[row] = accum;
  }
}

void Test(KernelFunc func, const int matrix_height, const int vector_height,
          dim3 grid, dim3 block) {
  thrust::device_vector<float> a(matrix_height * vector_height);
  thrust::device_vector<float> b(vector_height);
  thrust::device_vector<float> c(vector_height);
  constexpr int mat_fill_value = 3;
  constexpr int vec_fill_value = 2;

  thrust::fill(a.begin(), a.end(), mat_fill_value);
  thrust::fill(b.begin(), b.end(), vec_fill_value);
  thrust::fill(c.begin(), c.end(), 0);

  float *a_ptr = thrust::raw_pointer_cast(a.data());
  float *b_ptr = thrust::raw_pointer_cast(b.data());
  float *c_ptr = thrust::raw_pointer_cast(c.data());

  func<<<grid, block>>>(a_ptr, b_ptr, c_ptr, vector_height);
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  auto host_c = thrust::host_vector<float>(c);

  float *host_c_ptr = thrust::raw_pointer_cast(host_c.data());

  const int anwser = mat_fill_value * vec_fill_value * vector_height;
  for (const int row : std::views::iota(0, vector_height)) {
    const auto index = row;
    if (host_c_ptr[index] != anwser) {
      std::string error_string = "Houston we have a problem!\n";
      error_string += fmt::format("At ({},{}) found value: {} instead of {}!\n", row, 1, host_c_ptr[index], anwser);
      std::cout<<error_string;
      exit(1);
    }
  }
  std::cout << "All good brother!\n";
}

int main() {
  // Mat x Vec
  constexpr int mat_height = 7328;
  constexpr int vec_height = 4016;
  constexpr int block_size = 32;
  dim3 grid(ceil_div(vec_height, block_size));
  dim3 block(block_size);

  Test(MatrixVectorKernel, mat_height, vec_height, grid, block);

  // profile the relevant kernels:
  // ncu -k "regex:Matrix" ./bin/matrix_mul_variants
  return 0;
}