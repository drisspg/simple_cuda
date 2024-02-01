
#include "src/include/utils.h"
#include <cmath>
#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <fmt/core.h>

using KernelFunc = void (*)(float *, float *, float *, int);

__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width) {
  // Two sqaure matrices and performs matmul
  const auto row_stride = Width;
  const auto col_stride = 1;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (row < Width && col < Width) {
    float accum = 0;
    for (int k = 0; k < Width; k++) {
      int m_idx = row * row_stride + k * col_stride;
      int n_idx = k * row_stride + col * col_stride;
      accum += M[m_idx] * N[n_idx];
    }
    P[row * row_stride + col * col_stride] = accum;
  }
}

__global__ void MatrixMulRow(float *M, float *N, float *P, int Width) {
  // Two square matrices, 1 thread produces a row of the output
  const auto row_stride = Width;
  const auto col_stride = 1;
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (row < Width) {
    for (int col = 0; col < Width; col++) {
      float accum = 0;
      for (int k = 0; k < Width; k++) {
        int m_idx = row * row_stride + k * col_stride;
        int n_idx = k * row_stride + col * col_stride;
        accum += M[m_idx] * N[n_idx];
      }
      P[row * row_stride + col * col_stride] = accum;
    }
  }
}


__global__ void MatrixMulCol(float *M, float *N, float *P, int Width) {
  // Two square matrices, 1 thread produces a row of the output
  const auto row_stride = Width;
  const auto col_stride = 1;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (col < Width) {
    for (int row = 0; row < Width; row++) {
      float accum = 0;
      for (int k = 0; k < Width; k++) {
        int m_idx = row * row_stride + k * col_stride;
        int n_idx = k * row_stride + col * col_stride;
        accum += M[m_idx] * N[n_idx];
      }
      P[row * row_stride + col * col_stride] = accum;
    }
  }
}

void Test(KernelFunc func, const int width, dim3 grid, dim3 block) {
  thrust::device_vector<float> a(pow(width, 2));
  thrust::device_vector<float> b(pow(width, 2));
  thrust::device_vector<float> c(pow(width, 2));

  thrust::fill(a.begin(), a.end(), 1);
  thrust::fill(b.begin(), b.end(), 2);
  thrust::fill(c.begin(), c.end(), 0);

  float *a_ptr = thrust::raw_pointer_cast(a.data());
  float *b_ptr = thrust::raw_pointer_cast(b.data());
  float *c_ptr = thrust::raw_pointer_cast(c.data());

  func<<<grid, block>>>(a_ptr, b_ptr, c_ptr, width);
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  auto host_c = thrust::host_vector<float>(c);

  float *host_c_ptr = thrust::raw_pointer_cast(host_c.data());

  for (const int row : std::views::iota(0, width)) {
    for (const int col : std::views::iota(0, width)) {
      const auto index = row * width + col;
      if (host_c_ptr[index] != 2 * width) {
        std::string error_string = "Houston we have a problem!\n";
        error_string += fmt::format("At ({},{}) found value: {} instead of {}!\n", row, 1, host_c_ptr[index], 2 * width);
        std::cout<<error_string;
        exit(1);
      }
    }
  }
  std::cout << "All good brother!\n";
}

int main() {
  // Standard Matmul
  constexpr int width = 4096;
  constexpr int block_size = 32;
  dim3 grid(ceil_div(4096, block_size), ceil_div(4096, block_size));
  dim3 block(block_size, block_size);

  Test(MatrixMulKernel, width, grid, block);

  // RowMatmul
  grid = (ceil_div(4096, block_size));
  block = (block_size);
  Test(MatrixMulRow, width, grid, block);

// ColMatmul
  grid = (ceil_div(4096, block_size));
  block = (block_size);
  Test(MatrixMulCol, width, grid, block);

// profile the relevant kernels:
// ncu -k "regex:Matrix" ./bin/matrix_mul_variants
  return 0;
}