// Copyright © 2024 Apple Inc.

#include <cassert>
#include <iostream>

#include "mlx/mlx.h"

using namespace mlx::core;

int main() {
  // To use Metal debugging and profiling:
  // 1. Build with the MLX_METAL_DEBUG CMake option (i.e. -DMLX_METAL_DEBUG=ON).
  // 2. Run with MTL_CAPTURE_ENABLED=1.
  metal::start_capture("fft23_mlx_trace.gputrace");

  int n = 23;
  // int batch_size = 1;
  int batch_size = 131072 * 1024 / n;
  // array x = tile(expand_dims(arange(n), 0), {batch_size, 1});
  array x = random::normal({batch_size, n}) +
      complex64_t{0.0f, 1.0f} * random::normal({batch_size, n});
  x = astype(x, complex64);
  array y = fft::fft(x);

  std::cout << y << std::endl;

  metal::stop_capture();
}
