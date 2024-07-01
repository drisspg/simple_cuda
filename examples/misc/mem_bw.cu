#include <cooperative_groups.h>
#include <fmt/core.h>

using namespace cooperative_groups;

__global__ void direct_copy_optimized(int4 *output, int4 *input, size_t n) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < n / 4; i += stride) {
        output[i] = input[i];
    }
}

bool check_equal(int *output, int *input, int n) {
  for (int i = 0; i < n; i++) {
    if (output[i] != input[i]) {
      fmt::print("Not equal for {}, input: {} output: {}\n", i, input[i], output[i]);
      return false;
    }
  }
  return true;
}

int main() {

  int n = 1 << 24;
  int blockSize = 1024;
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  //  manual Grid_size
  int nBlocks_manual = 32 * numSMs;
  int *output, *data;
  cudaMallocManaged(&output, n * sizeof(int));
  cudaMallocManaged(&data, n * sizeof(int));
  std::fill_n(data, n, 1); // initialize data

  direct_copy_optimized<<<nBlocks_manual, blockSize>>>(reinterpret_cast<int4*>(output), reinterpret_cast<int4*>(data), n);
  cudaDeviceSynchronize();

  auto eq = check_equal(output, data, n);
  fmt::print("Equal: {}\n", eq);

  return 0;
}
