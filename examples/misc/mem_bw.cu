#include <cooperative_groups.h>
#include <fmt/core.h>
#include "utils.h"
#include <chrono>  // Add this for timing

using namespace cooperative_groups;

__global__ void direct_copy_optimized(float4 *output, float4 *input, size_t n) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < n / 4; i += stride) {
        output[i] = input[i];
    }
}

bool check_equal(float *output, float *input, int n) {
  for (int i = 0; i < n; i++) {
    if (output[i] != input[i]) {
      fmt::print("Not equal for {}, input: {} output: {}\n", i, input[i], output[i]);
      return false;
    }
  }
  return true;
}

int main() {
  int n = 1 << 28;
  int blockSize = 256;
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  //  manual Grid_size
  float nBlocks_manual = min(1024 * numSMs, simple_cuda::ceil_div(n, blockSize));
  float *output, *data;
  cudaMallocManaged(&output, n * sizeof(float));
  cudaMallocManaged(&data, n * sizeof(float));
  std::fill_n(data, n, 1); // initialize data

  // Make sure data is on the device before timing
  cudaMemPrefetchAsync(data, n * sizeof(float), 0);
  cudaDeviceSynchronize();

  direct_copy_optimized<<<nBlocks_manual, blockSize>>>(reinterpret_cast<float4*>(output), reinterpret_cast<float4*>(data), n);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start time
  cudaEventRecord(start);

  // Launch kernel
  direct_copy_optimized<<<nBlocks_manual, blockSize>>>(reinterpret_cast<float4*>(output), reinterpret_cast<float4*>(data), n);

  // Record stop time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Calculate elapsed time in milliseconds
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Clean up events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Check results
  auto eq = check_equal(output, data, n);
  fmt::print("Equal: {}\n", eq);

  // Calculate bandwidth
  auto read_bytes = n * sizeof(float);
  auto write_bytes = n * sizeof(float);
  auto total_bytes = read_bytes + write_bytes;

  // Convert to TB/s (terabytes per second)
  // 1 TB = 1024^4 bytes, but often simplified as 10^12 bytes
  double seconds = milliseconds / 1000.0;
  double gigabytes = total_bytes / 1e9;  // Convert bytes to GB
  double terabytes = gigabytes / 1000.0; // Convert GB to TB
  double bandwidth_TBs = terabytes / seconds;

  fmt::print("Time: {:.3f} ms\n", milliseconds);
  fmt::print("Data size: {:.2f} GB\n", gigabytes);
  fmt::print("Bandwidth: {:.2f} GB/s\n", gigabytes / seconds);
  fmt::print("Bandwidth: {:.2f} TB/s\n", bandwidth_TBs);

  // Free memory
  cudaFree(output);
  cudaFree(data);

  return 0;
}
