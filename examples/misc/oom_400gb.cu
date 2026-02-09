#include <cuda_runtime.h>
#include <fmt/core.h>
#include <cstdlib>
#include <cstdint>

int main() {
  constexpr std::size_t gib = 1024ULL * 1024ULL * 1024ULL;
  constexpr std::size_t requested_bytes = 400ULL * gib;

  fmt::print("Requesting {} GiB ({} bytes)\n", requested_bytes / gib, requested_bytes);

  void* host_ptr = std::malloc(requested_bytes);
  if (host_ptr == nullptr) {
    fmt::print("malloc: OOM (nullptr)\n");
  } else {
    fmt::print("malloc: success (can still be overcommit)\n");
    std::free(host_ptr);
  }

  void* pinned_ptr = nullptr;
  cudaError_t pinned_alloc_result = cudaMallocHost(&pinned_ptr, requested_bytes);
  if (pinned_alloc_result == cudaSuccess) {
    fmt::print("cudaMallocHost: success\n");
    cudaFreeHost(pinned_ptr);
  } else {
    fmt::print(
        "cudaMallocHost: {} ({})\n",
        cudaGetErrorName(pinned_alloc_result),
        cudaGetErrorString(pinned_alloc_result));
  }

  return 0;
}
