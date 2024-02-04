
#include "src/include/utils.h"
#include <cmath>
#include <fmt/core.h>
#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using KernelFunc = void (*)(float *, float *, float *, int);
using namespace simple_cuda;
struct Strides {
  int row_stride;
  int col_stride;
};

/**
 * This function will fill the shared memory tile with values from M and N matrices
 *
 * The M tile slides along the cols and the N tile slides down rows
 */
template <int tile_size>
__device__ void fill_tiles(float *m_tile, float *n_tile, float *M, float *N,
                           const int tile_idx, const int row, const int col,
                           const Strides stride, const int width) {
  const auto row_stride = stride.row_stride;
  const auto col_stride = stride.col_stride;

  const int tile_offset = tile_idx * tile_size;
  const int global_m_col = tile_offset + threadIdx.x;
  const int global_n_row = tile_offset + threadIdx.y;

  const int m_idx = row * row_stride + global_m_col * col_stride;
  const int n_idx = col * col_stride + global_n_row * row_stride;

  m_tile[threadIdx.y * tile_size + threadIdx.x] =
      global_m_col < width ? M[m_idx] : 0.0;
  n_tile[threadIdx.y * tile_size + threadIdx.x] =
      global_n_row < width ? N[n_idx] : 0.0;

}

template <int tile_size>
__global__ void MatrixMulKernelTiled(float *M, float *N, float *P, int width) {
  // Two sqaure matrices and performs matmul
  const Strides matrix_strides{width, 1};
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

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
  if (row < width and col < width) {
    const int out_idx =
        row * matrix_strides.row_stride + col * matrix_strides.col_stride;

    P[out_idx] = accumulator;
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

  const float anwser = 2 * width;
  for (const int row : std::views::iota(0, width)) {
    for (const int col : std::views::iota(0, width)) {
      const auto index = row * width + col;
      if (host_c_ptr[index] != anwser) {
        std::string error_string = "Houston we have a problem!\n";
        error_string +=
            fmt::format("At ({},{}) found value: {} instead of {}!\n", row, 1,
                        host_c_ptr[index], anwser);
        std::cout << error_string;
        exit(1);
      }
    }
  }
  std::cout << "All good brother!\n";
}

int main() {
  // Standard Matmul
  constexpr int width = 8001;
  constexpr int block_size = 32;

  dim3 grid(ceil_div(width, block_size), ceil_div(width, block_size));
  dim3 block(block_size, block_size);

  Test(MatrixMulKernelTiled<block_size>, width, grid, block);

  // profile the relevant kernels:
  // ncu -k "regex:Matrix" ./bin/tile_mm
  return 0;
}