#include <cooperative_groups.h>
#include "src/include/utils.h"
#include <fmt/core.h>

using namespace cooperative_groups;
__device__ int reduce_sum(thread_group g, int *temp, int val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane<i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__device__ int thread_sum(int *input, int n) {
  int sum = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4;
       i += blockDim.x * gridDim.x) {
    int4 in = ((int4 *)input)[i];
    sum += in.x + in.y + in.z + in.w;
  }
  return sum;
}

__global__ void sum_kernel_block(int *sum, int *input, int n) {
  int my_sum = thread_sum(input, n);

  extern __shared__ int temp[];
  auto g = this_thread_block();
  int block_sum = reduce_sum(g, temp, my_sum);

  if (g.thread_rank() == 0)
    atomicAdd(sum, block_sum);
}

int main() {

  int n = 1 << 24;
  int blockSize = 128;
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  constexpr int coarse_factor = 32;
  //  manual Grid_size
  int nBlocks_manual = 64 * numSMs;
  int nBlocks = simple_cuda::ceil_div(n, blockSize *  coarse_factor * 2);
  int sharedBytes = blockSize * sizeof(int);

  int *sum, *data;
  cudaMallocManaged(&sum, sizeof(int));
  cudaMallocManaged(&data, n * sizeof(int));
  std::fill_n(data, n, 1); // initialize data
  cudaMemset(sum, 0, sizeof(int));

  sum_kernel_block<<<nBlocks_manual, blockSize, sharedBytes>>>(sum, data, n);
  cudaDeviceSynchronize();
  fmt::print("The array is sized {}\n", n);
  fmt::print("Sum is equal to {}\n", *sum);

  return 0;
}
