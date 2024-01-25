#include "../src/include/utils.h"
#include <stdio.h>
// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

const int DSIZE = 8192;
const int block_size = 16; // CUDA maximum is 1024 *total* threads in block
const float A_val = 1.0f;
const float B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  int row_idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
  int col_idx = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

  if ((row_idx < ds) && (col_idx < ds)) {
    float temp = 0;
    for (int i = 0; i < ds; i++) {
      int row_stride = ds;
      int col_sride = 1;
      int a_index = row_idx * row_stride + i * col_sride;
      int b_index = col_idx * col_sride + i * row_stride;
      temp += A[a_index] * B[b_index]; // dot product of row and column
    }
    C[row_idx * ds + col_idx] = temp;
  }
}

int main() {

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;
  // start timing
  t0 = clock();

  h_A = new float[DSIZE * DSIZE];
  h_B = new float[DSIZE * DSIZE];
  h_C = new float[DSIZE * DSIZE];
  for (int i = 0; i < DSIZE * DSIZE; i++) {
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;
  }

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size); // dim3 variable holds 3 dimensions
  dim3 grid(ceil_div(DSIZE, block.x), ceil_div(DSIZE, block.y));
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE * DSIZE; i++)
    if (h_C[i] != A_val * B_val * DSIZE) {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i],
             A_val * B_val * DSIZE);
      return -1;
    }
  printf("Success!\n");

  return 0;
}
