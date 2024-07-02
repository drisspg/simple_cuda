#include <cooperative_groups.h>
#include <fmt/core.h>
#include "utils.h"

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

  direct_copy_optimized<<<nBlocks_manual, blockSize>>>(reinterpret_cast<float4*>(output), reinterpret_cast<float4*>(data), n);
  cudaDeviceSynchronize();

  auto eq = check_equal(output, data, n);
  fmt::print("Equal: {}\n", eq);

  return 0;
}
