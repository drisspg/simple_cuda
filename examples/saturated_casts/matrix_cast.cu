#include "src/include/utils.h"
#include <cmath>
#include <fmt/core.h>
#include <ranges>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

using namespace simple_cuda;

int main() {
  // Standard Matmul
  constexpr int width = 8001;
  constexpr int block_size = 32;

  dim3 grid(ceil_div(width, block_size), ceil_div(width, block_size));
  dim3 block(block_size, block_size);

  // profile the relevant kernels:
  // ncu -k "regex:Matrix" ./bin/tile_mm
  return 0;
}