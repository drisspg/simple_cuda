#include <cuda_fp8.h>
#include <stdio.h>

__global__ void convert_to_e8m0(float *in, __nv_fp8_storage_t *out) {
  const float input_val = in[0];
  printf("Device input value: %f\n", input_val);
  __nv_fp8_storage_t result =
      __nv_cvt_float_to_e8m0(input_val, __NV_SATFINITE, cudaRoundNearest);
  printf("Device output value (hex): 0x%02x, (decimal): %u\n",
         (unsigned char)result, (unsigned char)result);
  out[0] = result;
}

int main() {
  float h_in = 1.0f / 448.0f;
  float *d_in;
  __nv_fp8_storage_t *d_out, h_out;

  cudaMalloc(&d_in, sizeof(float));
  cudaMalloc(&d_out, sizeof(__nv_fp8_storage_t));

  cudaError_t err =
      cudaMemcpy(d_in, &h_in, sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("Memcpy error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  convert_to_e8m0<<<1, 1>>>(d_in, d_out);
  cudaDeviceSynchronize(); // Need this to see printf from kernel
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  err = cudaMemcpy(&h_out, d_out, sizeof(__nv_fp8_storage_t),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("Memcpy error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  printf("Host input float: %f\n", h_in);
  printf("Host output e8m0 hex: 0x%02x, decimal: %u\n", (unsigned char)h_out,
         (unsigned char)h_out);
  printf("Host output e8m0 bits: ");
  for (int i = 7; i >= 0; i--) {
    printf("%d", (h_out >> i) & 0x1);
  }
  printf("\n");

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
