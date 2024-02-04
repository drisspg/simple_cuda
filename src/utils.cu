#include "include/utils.h"

using namespace simple_cuda;

float kernel_time(std::function<void()> kernelLauncher) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  kernelLauncher();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsedTime;
}