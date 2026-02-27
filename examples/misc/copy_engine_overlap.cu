#include <cuda_runtime.h>
#include <fmt/core.h>
#include "src/include/utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace {

__global__ void transform_kernel(float* output, const float* input, int count, int rounds) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  float value = input[index];
  for (int iter = 0; iter < rounds; ++iter) {
    value = fmaf(value, 1.000001f, 0.000001f);
  }
  output[index] = value;
}

double run_serial_pipeline(float* host_output,
                           const float* host_input,
                           std::size_t total_elements,
                           std::size_t chunk_elements,
                           int rounds,
                           float* device_input,
                           float* device_output,
                           cudaStream_t stream) {
  auto start = std::chrono::high_resolution_clock::now();
  std::size_t chunk_count = (total_elements + chunk_elements - 1) / chunk_elements;

  for (std::size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    std::size_t element_offset = chunk_index * chunk_elements;
    std::size_t current_elements = std::min(chunk_elements, total_elements - element_offset);
    std::size_t bytes = current_elements * sizeof(float);

    CUDA_TRY(cudaMemcpyAsync(device_input, host_input + element_offset, bytes, cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = static_cast<int>((current_elements + threads - 1) / threads);
    transform_kernel<<<blocks, threads, 0, stream>>>(device_output, device_input, static_cast<int>(current_elements), rounds);
    CUDA_TRY(cudaGetLastError());

    CUDA_TRY(cudaMemcpyAsync(host_output + element_offset, device_output, bytes, cudaMemcpyDeviceToHost, stream));
  }

  CUDA_TRY(cudaStreamSynchronize(stream));
  auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

double run_overlapped_pipeline(float* host_output,
                               const float* host_input,
                               std::size_t total_elements,
                               std::size_t chunk_elements,
                               int rounds,
                               const std::vector<float*>& device_inputs,
                               const std::vector<float*>& device_outputs,
                               cudaStream_t h2d_stream,
                               cudaStream_t compute_stream,
                               cudaStream_t d2h_stream) {
  std::size_t pipeline_depth = device_inputs.size();
  std::vector<cudaEvent_t> h2d_done(pipeline_depth);
  std::vector<cudaEvent_t> compute_done(pipeline_depth);
  std::vector<cudaEvent_t> d2h_done(pipeline_depth);
  for (std::size_t buffer = 0; buffer < pipeline_depth; ++buffer) {
    CUDA_TRY(cudaEventCreateWithFlags(&h2d_done[buffer], cudaEventDisableTiming));
    CUDA_TRY(cudaEventCreateWithFlags(&compute_done[buffer], cudaEventDisableTiming));
    CUDA_TRY(cudaEventCreateWithFlags(&d2h_done[buffer], cudaEventDisableTiming));
    CUDA_TRY(cudaEventRecord(d2h_done[buffer], d2h_stream));
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::size_t chunk_count = (total_elements + chunk_elements - 1) / chunk_elements;

  for (std::size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    std::size_t buffer = chunk_index % pipeline_depth;
    std::size_t element_offset = chunk_index * chunk_elements;
    std::size_t current_elements = std::min(chunk_elements, total_elements - element_offset);
    std::size_t bytes = current_elements * sizeof(float);

    CUDA_TRY(cudaStreamWaitEvent(h2d_stream, d2h_done[buffer], 0));
    CUDA_TRY(cudaMemcpyAsync(device_inputs[buffer], host_input + element_offset, bytes, cudaMemcpyHostToDevice, h2d_stream));
    CUDA_TRY(cudaEventRecord(h2d_done[buffer], h2d_stream));

    CUDA_TRY(cudaStreamWaitEvent(compute_stream, h2d_done[buffer], 0));
    int threads = 256;
    int blocks = static_cast<int>((current_elements + threads - 1) / threads);
    transform_kernel<<<blocks, threads, 0, compute_stream>>>(
        device_outputs[buffer], device_inputs[buffer], static_cast<int>(current_elements), rounds);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaEventRecord(compute_done[buffer], compute_stream));

    CUDA_TRY(cudaStreamWaitEvent(d2h_stream, compute_done[buffer], 0));
    CUDA_TRY(cudaMemcpyAsync(host_output + element_offset, device_outputs[buffer], bytes, cudaMemcpyDeviceToHost, d2h_stream));
    CUDA_TRY(cudaEventRecord(d2h_done[buffer], d2h_stream));
  }

  CUDA_TRY(cudaStreamSynchronize(d2h_stream));
  auto stop = std::chrono::high_resolution_clock::now();

  for (std::size_t buffer = 0; buffer < pipeline_depth; ++buffer) {
    CUDA_TRY(cudaEventDestroy(h2d_done[buffer]));
    CUDA_TRY(cudaEventDestroy(compute_done[buffer]));
    CUDA_TRY(cudaEventDestroy(d2h_done[buffer]));
  }

  return std::chrono::duration<double, std::milli>(stop - start).count();
}

bool sampled_match(const float* first, const float* second, std::size_t count) {
  std::size_t stride = std::max<std::size_t>(1, count / 1024);
  for (std::size_t index = 0; index < count; index += stride) {
    if (std::abs(first[index] - second[index]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

}

int main(int argc, char** argv) {
  cudaDeviceProp properties{};
  CUDA_TRY(cudaGetDeviceProperties(&properties, 0));

  std::size_t total_elements = 1ULL << 26;
  std::size_t chunk_elements = 1ULL << 22;
  std::size_t chunk_bytes = chunk_elements * sizeof(float);
  std::size_t total_bytes = total_elements * sizeof(float);
  int rounds = 96;
  int pipeline_depth = 2;
  if (argc > 1) {
    pipeline_depth = std::atoi(argv[1]);
  }
  if (pipeline_depth < 1) {
    fmt::print(stderr, "pipeline_depth must be >= 1\n");
    return 1;
  }
  std::size_t overlap_buffers = static_cast<std::size_t>(pipeline_depth);

  fmt::print("Device: {}\n", properties.name);
  fmt::print("asyncEngineCount: {}\n", properties.asyncEngineCount);
  fmt::print("total elements: {}\n", total_elements);
  fmt::print("chunk elements: {}\n", chunk_elements);
  fmt::print("compute rounds: {}\n\n", rounds);
  fmt::print("pipeline depth: {}\n\n", pipeline_depth);

  float* host_input = nullptr;
  float* host_serial_output = nullptr;
  float* host_overlapped_output = nullptr;
  CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&host_input), total_bytes));
  CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&host_serial_output), total_bytes));
  CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&host_overlapped_output), total_bytes));

  for (std::size_t index = 0; index < total_elements; ++index) {
    host_input[index] = static_cast<float>(index % 2048) / 2048.0f;
  }

  float* serial_device_input = nullptr;
  float* serial_device_output = nullptr;
  std::vector<float*> overlapped_device_inputs(overlap_buffers, nullptr);
  std::vector<float*> overlapped_device_outputs(overlap_buffers, nullptr);
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&serial_device_input), chunk_bytes));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&serial_device_output), chunk_bytes));
  for (std::size_t buffer = 0; buffer < overlap_buffers; ++buffer) {
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&overlapped_device_inputs[buffer]), chunk_bytes));
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&overlapped_device_outputs[buffer]), chunk_bytes));
  }

  cudaStream_t serial_stream;
  cudaStream_t h2d_stream;
  cudaStream_t compute_stream;
  cudaStream_t d2h_stream;
  CUDA_TRY(cudaStreamCreateWithFlags(&serial_stream, cudaStreamNonBlocking));
  CUDA_TRY(cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking));
  CUDA_TRY(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
  CUDA_TRY(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking));

  CUDA_TRY(cudaDeviceSynchronize());
  double serial_ms = run_serial_pipeline(host_serial_output,
                                         host_input,
                                         total_elements,
                                         chunk_elements,
                                         rounds,
                                         serial_device_input,
                                         serial_device_output,
                                         serial_stream);

  CUDA_TRY(cudaDeviceSynchronize());
  double overlapped_ms = run_overlapped_pipeline(host_overlapped_output,
                                                 host_input,
                                                 total_elements,
                                                 chunk_elements,
                                                 rounds,
                                                 overlapped_device_inputs,
                                                 overlapped_device_outputs,
                                                 h2d_stream,
                                                 compute_stream,
                                                 d2h_stream);

  bool outputs_match = sampled_match(host_serial_output, host_overlapped_output, total_elements);
  double moved_gb = (2.0 * static_cast<double>(total_bytes)) / 1e9;
  double serial_gbps = moved_gb / (serial_ms / 1000.0);
  double overlapped_gbps = moved_gb / (overlapped_ms / 1000.0);

  fmt::print("serial pipeline:    {:8.3f} ms  {:8.2f} GB/s\n", serial_ms, serial_gbps);
  fmt::print("overlapped pipeline:{:8.3f} ms  {:8.2f} GB/s\n", overlapped_ms, overlapped_gbps);
  fmt::print("speedup:            {:8.3f}x\n", serial_ms / overlapped_ms);
  fmt::print("sampled match:      {}\n", outputs_match ? "true" : "false");

  CUDA_TRY(cudaStreamDestroy(serial_stream));
  CUDA_TRY(cudaStreamDestroy(h2d_stream));
  CUDA_TRY(cudaStreamDestroy(compute_stream));
  CUDA_TRY(cudaStreamDestroy(d2h_stream));
  CUDA_TRY(cudaFree(serial_device_input));
  CUDA_TRY(cudaFree(serial_device_output));
  for (std::size_t buffer = 0; buffer < overlap_buffers; ++buffer) {
    CUDA_TRY(cudaFree(overlapped_device_inputs[buffer]));
    CUDA_TRY(cudaFree(overlapped_device_outputs[buffer]));
  }
  CUDA_TRY(cudaFreeHost(host_input));
  CUDA_TRY(cudaFreeHost(host_serial_output));
  CUDA_TRY(cudaFreeHost(host_overlapped_output));

  return outputs_match ? 0 : 1;
}
