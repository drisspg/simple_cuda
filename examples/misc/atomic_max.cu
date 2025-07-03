#include <cuda_runtime.h>
#include <iostream>
#include <limits>

__device__ __forceinline__ void atomicMaxFloatBAD(float *addr, float value) {
  // source: https://stackoverflow.com/a/51549250
   (value >= 0)
             ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
             : __uint_as_float(
                   atomicMin((unsigned int *)addr, __float_as_uint(value)));
}

__device__ __forceinline__ float atomicMaxFloatGOOD(float *addr, float value) {
  // source: https://stackoverflow.com/a/51549250

  return !signbit(value)
             ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
             : __uint_as_float(
                   atomicMin((unsigned int *)addr, __float_as_uint(value)));
}

__device__ static float atomicMaxCAS(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__ void compareSubnormalToNegInf(float *data) {
  // Assuming blockDim.x * gridDim.x >= 2
  float subnormal = 1.4013e-45; // Smallest positive
  if (threadIdx.x == 0) {
    atomicMaxFloatBAD(&data[0], subnormal);
  } else if (threadIdx.x == 1) {
    atomicMaxFloatGOOD(&data[1], subnormal);
  } else if (threadIdx.x == 2){
    atomicMaxCAS(&data[2], subnormal);
  }
}

int main() {
  float neg_inf = -std::numeric_limits<float>::infinity();
  float *d_data, h_data[3] = {neg_inf, neg_inf, neg_inf};

  cudaMalloc(&d_data, 3 * sizeof(float));
  cudaMemcpy(d_data, h_data, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  compareSubnormalToNegInf<<<1, 2>>>(d_data);
  cudaDeviceSynchronize();

  cudaMemcpy(h_data, d_data, 3 * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Subnormal atomicMaxBad result: " << h_data[0] << std::endl;
  std::cout << "Subnormal atomicMaxGOOD result: " << h_data[1] << std::endl;
std::cout << "Subnormal atomicMaxCAS result: " << h_data[2] << std::endl;

  cudaFree(d_data);

  return 0;
}
