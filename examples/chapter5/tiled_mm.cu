
#include "src/include/utils.h"
#include <cmath>
#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using KernelFunc = void (*)(float *, float *, float *, int);

struct Strides {
  int row_stride;
  int col_stride;
};

template <int tile_size>
__device__ void fill_tiles(float *m_tile, float *n_tile, float *M, float *N,
                           const int tile_idx, const int row, const int col,
                           const Strides stride, const int width) {
  const auto row_stride = stride.row_stride;
  const auto col_stride = stride.col_stride;

  const int m_idx = row * row_stride + tile_idx * tile_size * col_stride +
                    threadIdx.x * col_stride;
  const int n_idx = col * col_stride + tile_idx * tile_size * row_stride +
                    threadIdx.y * row_stride;

  m_tile[threadIdx.y * tile_size + threadIdx.x] =
      m_idx < width * width ? M[m_idx] : 0.0;
  n_tile[threadIdx.y * tile_size + threadIdx.x] =
      n_idx < width * width ? N[n_idx] : 0.0;
}

template <int tile_size>
__global__ void MatrixMulKernelTiled(float *M, float *N, float *P, int width) {
  // Two sqaure matrices and performs matmul
  const Strides matrix_strides{width, 1};
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row > width || col > width)
    return;

  __shared__ float M_tile[tile_size * tile_size];
  __shared__ float N_tile[tile_size * tile_size];

  const int num_tiles = ceil_div(width, tile_size);

  float accumulator = 0.0;
  for (int tile{0}; tile < num_tiles; tile++) {
    fill_tiles<tile_size>(M_tile, N_tile, M, N, tile, row, col, matrix_strides,
                          width);
    __syncthreads();
    for (int k{0}; k < tile_size; k++) {
      accumulator += M_tile[threadIdx.y * tile_size + k] *
                     N_tile[k * tile_size + threadIdx.x];
    }
    __syncthreads();
  }
  const int out_idx =
      row * matrix_strides.row_stride + col * matrix_strides.col_stride;

  P[out_idx] = accumulator;
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
        std::cout << "houston we have a problem!\n";
        std::cout << "At (" << row << "," << col
                  << ") found value: " << host_c_ptr[index] << std::endl;
        exit(1);
      }
    }
  }
  std::cout << "All good brother!\n";
}

int main() {
  // Standard Matmul
  constexpr int width = 8192;
  constexpr int block_size = 32;

  dim3 grid(ceil_div(width, block_size), ceil_div(width, block_size));
  dim3 block(block_size, block_size);

  Test(MatrixMulKernelTiled<block_size>, width, grid, block);

  // profile the relevant kernels:
  // ncu -k "regex:Matrix" ./bin/matrix_mul_variants
  return 0;
}