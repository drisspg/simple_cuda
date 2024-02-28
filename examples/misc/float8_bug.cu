#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <functional>
#include <stdio.h>
#include <type_traits>
#include <fmt/core.h>
#include <bitset>
#include <iostream>

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


__global__ void saturated_cast_kernel_single(
   const float *input, __nv_fp8_storage_t *output, int n_rows, int n_cols,
    __nv_fp8_interpretation_t out_dtype, float *scaler) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Assume row major
  const int global_index = row * n_cols + col;
  if (row < n_rows && col < n_cols) {
      const float scaled_input = input[global_index] * (*scaler);
      output[global_index] = __nv_cvt_float_to_fp8(
          scaled_input, __nv_saturation_t::__NV_SATFINITE, out_dtype);
  }
}

int main() {
  float inpt[2] = {0.3223, 0.3223};
  float scale = 57344.0;
  float output[2] = {0.0, 0.0};

  // Pointer to device array
  float *d_inpt = nullptr;
  float *d_scale = nullptr;
  __nv_fp8_storage_t *d_output = nullptr;
  // Allocate memory on the device
  cudaMalloc((void **)&d_inpt, 2 * sizeof(float));
  cudaMalloc((void **)&d_scale, sizeof(float));
  cudaMalloc((void **)&d_output, 2 * sizeof(__nv_fp8_interpretation_t));
  cudaCheckErrors("cudaMalloc failure");

  cudaMemcpy(d_inpt, inpt, 2 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, &scale, sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  dim3 grid(1,1);
  dim3 block(2, 2);

  saturated_cast_kernel_single<<<grid, block>>>(
      static_cast<float *>(d_inpt), static_cast<__nv_fp8_storage_t *>(d_output),
      1, 2, __nv_fp8_interpretation_t::__NV_E5M2,
      static_cast<float *>(d_scale));
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(output, d_output, 2 * sizeof(__nv_fp8_storage_t),
             cudaMemcpyDeviceToHost);
  fmt::print("Output: {} {}\n", static_cast<uint8_t>(output[0]), static_cast<uint8_t>(output[1]));
  return 0;
}