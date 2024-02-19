
#include "src/include/utils.h"

#include <fmt/core.h>
#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using KernelFunc = void (*)(float *, float *, float *, int, int, int);
using namespace simple_cuda;
struct Strides {
  int row_stride;
  int col_stride;

  __host__ __device__ int index(const int row_num, const int col_num) const {
    return row_num * row_stride + col_num * col_stride;
  }
};

/**
 * This function will fill the shared memory tile with values from M and N
 * matrices
 *
 * The M tile slides along the cols and the N tile slides down rows
 */
template <int tile_size>
__device__ void fill_tiles(float *m_tile, float *n_tile, float *A, float *B,
                           const int tile_idx, const int row, const int col,
                           const Strides mat1_stride, const Strides mat2_stride,
                           const Strides tile_strides, const int inner_dim) {
  const int tile_offset = tile_idx * tile_size;

  const int global_m_col = tile_offset + threadIdx.x;
  const int global_n_row = tile_offset + threadIdx.y;

  const int A_index = mat1_stride.index(row, global_m_col);
  const int B_index = mat2_stride.index(global_n_row, col);

  const auto tile_index = tile_strides.index(threadIdx.y, threadIdx.x);
  m_tile[tile_index] = global_m_col < inner_dim ? A[A_index] : 0.0;
  n_tile[tile_index] = global_n_row < inner_dim ? B[B_index] : 0.0;
}

template <int tile_size>
__global__ void MatrixMulKernelTiled(float *Mat1, float *Mat2, float *OutMat,
                                     const int M, const int K, const int N) {
  // Two Row Major matrix multiplies outputing to row-major
  const Strides mat1_strides{K, 1};
  const Strides mat2_strides{N, 1};
  const Strides out_mat_strides{N, 1};

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float M_tile[tile_size * tile_size];
  __shared__ float N_tile[tile_size * tile_size];

  constexpr Strides tile_strides = {tile_size, 1};

  // The number of tiles to span the K dimension
  const int num_tiles = ceil_div(K, tile_size);

  float accumulator = 0.0;
  for (int tile{0}; tile < num_tiles; tile++) {
    fill_tiles<tile_size>(M_tile, N_tile, Mat1, Mat2, tile, row, col,
                          mat1_strides, mat2_strides, tile_strides, K);
    __syncthreads();
    for (int k{0}; k < tile_size; k++) {
      accumulator += M_tile[tile_strides.index(threadIdx.y, k)] *
                     N_tile[tile_strides.index(k, threadIdx.x)];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    const int out_idx = out_mat_strides.index(row, col);
    OutMat[out_idx] = accumulator;
  }
}

// Function to print the matrix
void printMatrix(const float *data, int M, int N, const Strides &strides) {
  fmt::print("Full matrix contents:\n");
  for (const int row : std::views::iota(0, M)) {
    for (const int col : std::views::iota(0, N)) {
      const auto index = strides.index(row, col);
      fmt::print("{} ", data[index]);
    }
    fmt::print("\n");
  }
}

void Test(KernelFunc func, const int M, const int K, const int N, dim3 grid,
          dim3 block) {
  thrust::device_vector<float> a(M * K);
  thrust::device_vector<float> b(K * N);
  thrust::device_vector<float> c(M * N);

  thrust::fill(a.begin(), a.end(), 1);
  thrust::fill(b.begin(), b.end(), 2);
  thrust::fill(c.begin(), c.end(), 0);

  float *a_ptr = thrust::raw_pointer_cast(a.data());
  float *b_ptr = thrust::raw_pointer_cast(b.data());
  float *c_ptr = thrust::raw_pointer_cast(c.data());

  func<<<grid, block>>>(a_ptr, b_ptr, c_ptr, M, K, N);
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();

  auto host_c = thrust::host_vector<float>(c);

  float *host_c_ptr = thrust::raw_pointer_cast(host_c.data());

  const Strides out_strides = {N, 1};
  const float anwser = 2 * K;
  for (const int row : std::views::iota(0, M)) {
    for (const int col : std::views::iota(0, N)) {
      const auto index = out_strides.index(row, col);
      if (host_c_ptr[index] != anwser) {
        std::string error_string = "Houston we have a problem!\n";
        error_string +=
            fmt::format("At ({},{}) found value: {} instead of {}!\n", row, col,
                        host_c_ptr[index], anwser);
        std::cout << error_string;
        if (M * K * N < 64) {
          printMatrix(host_c_ptr, M, N, out_strides);
        }
        exit(1);
      }
    }
  }
  std::cout << "All good brother!\n";
}

int main() {
  // Standard Matmul
  int M = 3;
  int K = 4;
  int N = 5;
  constexpr int block_size = 2;

  // dimx is inner dim, dimy is outerdim
  dim3 grid(ceil_div(N, block_size), ceil_div(M, block_size));
  dim3 block(block_size, block_size);

  Test(MatrixMulKernelTiled<block_size>, M, K, N, grid, block);

  M = 16;
  K = 12;
  N = 20;
  constexpr int block_size2 = 4;
  dim3 grid2(ceil_div(N, block_size2), ceil_div(M, block_size2));
  dim3 block2(block_size2, block_size2);
  Test(MatrixMulKernelTiled<block_size2>, M, K, N, grid2, block2);

  M = 1028;
  K = 19023;
  N = 2134;
  constexpr int block_size3= 32;
  dim3 grid3(ceil_div(N, block_size3), ceil_div(M, block_size3));
  dim3 block3(block_size3, block_size3);
  Test(MatrixMulKernelTiled<block_size3>, M, K, N, grid3, block3);

  // profile the relevant kernels:
  // ncu -k "regex:Matrix" ./bin/tile_mm
  return 0;
}