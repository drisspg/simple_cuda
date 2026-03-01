#include <algorithm>
#include <fmt/core.h>
#include <vector>
#include "utils.h"

__global__ void direct_copy_optimized(float4 *output,
                                      const float4 *input,
                                      size_t vec_n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < vec_n; i += stride) {
    output[i] = input[i];
  }
}

struct BenchResult {
  float ms;
  double gbps;
};

BenchResult bench_kernel(float4 *output,
                         float4 *input,
                         size_t vec_n,
                         int nBlocks,
                         int blockSize,
                         int iters,
                         double gigabytes) {
  direct_copy_optimized<<<nBlocks, blockSize>>>(output, input, vec_n);
  direct_copy_optimized<<<nBlocks, blockSize>>>(output, input, vec_n);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    direct_copy_optimized<<<nBlocks, blockSize>>>(output, input, vec_n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float ms = total_ms / iters;
  return {ms, gigabytes / (ms / 1000.0)};
}

BenchResult bench_memcpy(void *dst,
                         const void *src,
                         size_t bytes,
                         cudaMemcpyKind kind,
                         int iters,
                         double gigabytes) {
  cudaMemcpy(dst, src, bytes, kind);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    cudaMemcpy(dst, src, bytes, kind);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float ms = total_ms / iters;
  return {ms, gigabytes / (ms / 1000.0)};
}

void print_table_header() {
  fmt::print(
    "{:<12} {:<12} {:<22} {:<14} {:>10} {:>12} {:>10}\n",
    "src_space",
    "dst_space",
    "hints",
    "mechanism",
    "time_ms",
    "gbps",
    "roofline%");
  fmt::print(
    "{:-<12} {:-<12} {:-<22} {:-<14} {:-<10} {:-<12} {:-<10}\n",
    "",
    "",
    "",
    "",
    "",
    "",
    "");
}

void print_table_row(const char *src_space,
                     const char *dst_space,
                     const char *hints,
                     const char *mechanism,
                     const BenchResult &bench,
                     double peak_gbps) {
  fmt::print(
    "{:<12} {:<12} {:<22} {:<14} {:>10.3f} {:>12.2f} {:>10.1f}\n",
    src_space,
    dst_space,
    hints,
    mechanism,
    bench.ms,
    bench.gbps,
    100.0 * bench.gbps / peak_gbps);
}

int pick_blocks(size_t vec_n, int blockSize, int numSMs) {
  int maxBlocks = static_cast<int>(simple_cuda::ceil_div(
    vec_n, static_cast<size_t>(blockSize)));
  int blockCap = 1024 * numSMs;
  int nBlocks = std::min(maxBlocks, blockCap);
  return std::max(1, nBlocks);
}

int main() {
  size_t n = 1 << 28;
  size_t vec_n = n / 4;
  int blockSize = 256;
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  int nBlocks = pick_blocks(vec_n, blockSize, numSMs);
  const int iters = 20;
  constexpr double peak_gbps = 273.0;

  double total_bytes = n * sizeof(float) * 2.0;
  double gigabytes = total_bytes / 1e9;

  fmt::print(
    "config: block_size={} blocks={} iters={} total_bytes={:.2f} GB\n",
    blockSize,
    nBlocks,
    iters,
    gigabytes);
  print_table_header();

  {
    float *output, *data;
    cudaMallocManaged(&output, n * sizeof(float));
    cudaMallocManaged(&data, n * sizeof(float));
    std::fill_n(data, n, 1.0f);

    BenchResult bench = bench_kernel(
      reinterpret_cast<float4*>(output),
      reinterpret_cast<float4*>(data),
      vec_n,
      nBlocks,
      blockSize,
      iters,
      gigabytes);

    print_table_row(
      "managed", "managed", "none", "device-kernel", bench, peak_gbps);

    cudaFree(output);
    cudaFree(data);
  }

  {
    float *output, *data;
    cudaMallocManaged(&output, n * sizeof(float));
    cudaMallocManaged(&data, n * sizeof(float));
    std::fill_n(data, n, 1.0f);

    cudaMemLocation dev_loc{cudaMemLocationTypeDevice, 0};
    cudaMemAdvise(data, n * sizeof(float), cudaMemAdviseSetPreferredLocation, dev_loc);
    cudaMemAdvise(output, n * sizeof(float), cudaMemAdviseSetPreferredLocation, dev_loc);
    cudaMemPrefetchAsync(data, n * sizeof(float), dev_loc, 0, 0);
    cudaMemPrefetchAsync(output, n * sizeof(float), dev_loc, 0, 0);
    cudaDeviceSynchronize();

    BenchResult bench = bench_kernel(
      reinterpret_cast<float4*>(output),
      reinterpret_cast<float4*>(data),
      vec_n,
      nBlocks,
      blockSize,
      iters,
      gigabytes);

    print_table_row(
      "managed", "managed", "preferred+prefetch", "device-kernel", bench, peak_gbps);

    cudaFree(output);
    cudaFree(data);
  }

  {
    float *d_output, *d_data;
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMalloc(&d_data, n * sizeof(float));
    float *m_output, *m_data;
    cudaMallocManaged(&m_output, n * sizeof(float));
    cudaMallocManaged(&m_data, n * sizeof(float));

    std::vector<float> h_data(n, 1.0f);
    std::fill_n(m_data, n, 1.0f);
    std::fill_n(m_output, n, 0.0f);
    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    BenchResult bench = bench_kernel(
      reinterpret_cast<float4*>(d_output),
      reinterpret_cast<float4*>(d_data),
      vec_n,
      nBlocks,
      blockSize,
      iters,
      gigabytes);

    print_table_row(
      "device", "device", "-", "device-kernel", bench, peak_gbps);

    BenchResult d2d = bench_memcpy(
      d_output, d_data, n * sizeof(float), cudaMemcpyDeviceToDevice, iters, gigabytes);

    print_table_row(
      "device", "device", "-", "cudaMemcpy", d2d, peak_gbps);

    BenchResult managed_to_device = bench_memcpy(
      d_output, m_data, n * sizeof(float), cudaMemcpyDefault, iters, gigabytes);
    print_table_row(
      "managed", "device", "none", "cudaMemcpy", managed_to_device, peak_gbps);

    BenchResult device_to_managed = bench_memcpy(
      m_output, d_data, n * sizeof(float), cudaMemcpyDefault, iters, gigabytes);
    print_table_row(
      "device", "managed", "none", "cudaMemcpy", device_to_managed, peak_gbps);

    cudaMemLocation copy_hint_loc{cudaMemLocationTypeDevice, 0};
    cudaMemAdvise(m_data, n * sizeof(float), cudaMemAdviseSetPreferredLocation, copy_hint_loc);
    cudaMemAdvise(m_output, n * sizeof(float), cudaMemAdviseSetPreferredLocation, copy_hint_loc);
    cudaMemPrefetchAsync(m_data, n * sizeof(float), copy_hint_loc, 0, 0);
    cudaMemPrefetchAsync(m_output, n * sizeof(float), copy_hint_loc, 0, 0);
    cudaDeviceSynchronize();

    BenchResult managed_to_device_hinted = bench_memcpy(
      d_output, m_data, n * sizeof(float), cudaMemcpyDefault, iters, gigabytes);
    print_table_row(
      "managed", "device", "preferred+prefetch", "cudaMemcpy", managed_to_device_hinted, peak_gbps);

    BenchResult device_to_managed_hinted = bench_memcpy(
      m_output, d_data, n * sizeof(float), cudaMemcpyDefault, iters, gigabytes);
    print_table_row(
      "device", "managed", "preferred+prefetch", "cudaMemcpy", device_to_managed_hinted, peak_gbps);

    cudaFree(m_output);
    cudaFree(m_data);
    cudaFree(d_output);
    cudaFree(d_data);
  }

  fmt::print("theoretical_peak_gbps={:.1f} (LPDDR5X)\n", peak_gbps);
  return 0;
}
