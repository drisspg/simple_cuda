#include <algorithm>
#include <cstdlib>
#include <functional>
#include <random>
#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/generate.h>

namespace simple_cuda {

// error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "CUDA error: %s from %s at %s:%d\n", cudaGetErrorString(status), expression, file, line);
  std::exit(1);
}
#define CUDA_TRY(expr) ::simple_cuda::check_cuda((expr), #expr, __FILE__, __LINE__)
template <typename T, typename Y> T __host__ __device__ ceil_div(T a, Y b) {
  return a / b + (a % b != 0);
}

float kernel_time(std::function<void()> kernelLauncher);

template <typename T> void fill_random(T& input) {
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
  std::normal_distribution<float> dist{0, 1};
  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::generate(input.begin(), input.end(), gen);
}
} // namespace simple_cuda
