// Copyright © 2023 Apple Inc.

#include <numeric>
#include <set>

#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <iostream>

#define MAX_STOCKHAM_FFT_SIZE 2048
#define MAX_BLUESTEIN_FFT_SIZE 1024

namespace mlx::core::fft {

bool is_power_of_2(int n) {
  return ((n & (n - 1)) == 0) && n != 0;
}

int next_power_of_2(int n) {
  return pow(2, std::ceil(std::log2(n)));
}

array four_step_fft(
    const array& a,
    int n,
    int axis,
    bool real,
    bool inverse,
    StreamOrDevice s);

array fft_impl(
    const array& a,
    std::vector<int> n,
    const std::vector<int>& axes,
    bool real,
    bool inverse,
    StreamOrDevice s) {
  if (a.ndim() < 1) {
    throw std::invalid_argument(
        "[fftn] Requires array with at least one dimension.");
  }
  if (n.size() != axes.size()) {
    throw std::invalid_argument("[fftn] Shape and axes have different sizes.");
  }
  if (axes.empty()) {
    return a;
  }

  std::vector<size_t> valid_axes;
  for (int ax : axes) {
    valid_axes.push_back(ax < 0 ? ax + a.ndim() : ax);
  }
  std::set<int> unique_axes(valid_axes.begin(), valid_axes.end());
  if (unique_axes.size() != axes.size()) {
    std::ostringstream msg;
    msg << "[fftn] Duplicated axis received " << axes;
    throw std::invalid_argument(msg.str());
  }
  if (*unique_axes.begin() < 0 || *unique_axes.rbegin() >= a.ndim()) {
    std::ostringstream msg;
    msg << "[fftn] Invalid axis received for array with " << a.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  // In the following shape manipulations there are three cases to consider:
  // 1. In a complex to complex transform (fftn / ifftn) the output
  //    and input shapes are the same.
  // 2. In a real to complex transform (rfftn) n specifies the input dims
  //    and the output dims are n[i] / 2 + 1
  // 3  In a complex to real transform (irfftn) n specifies the output dims
  //    and the input dims are n[i] / 2 + 1

  if (std::any_of(n.begin(), n.end(), [](auto i) { return i <= 0; })) {
    std::ostringstream msg;
    msg << "[fftn] Invalid FFT output size requested " << n;
    throw std::invalid_argument(msg.str());
  }

  std::vector<int> in_shape = a.shape();
  for (int i = 0; i < valid_axes.size(); ++i) {
    in_shape[valid_axes[i]] = n[i];
  }
  if (real && inverse) {
    in_shape[valid_axes.back()] = n.back() / 2 + 1;
  }

  bool any_greater = false;
  bool any_less = false;
  for (int i = 0; i < in_shape.size(); ++i) {
    any_greater |= in_shape[i] > a.shape()[i];
    any_less |= in_shape[i] < a.shape()[i];
  }

  auto in = a;
  if (any_less) {
    in = slice(in, std::vector<int>(in.ndim(), 0), in_shape, s);
  }
  if (any_greater) {
    // Pad with zeros
    auto tmp = zeros(in_shape, a.dtype(), s);
    in = scatter(tmp, std::vector<array>{}, in, std::vector<int>{}, s);
  }

  auto out_shape = in_shape;
  if (real) {
    auto ax = valid_axes.back();
    out_shape[ax] = inverse ? n.back() : out_shape[ax] / 2 + 1;
  }

  auto stream = to_stream(s);

  auto in_type = real && !inverse ? float32 : complex64;
  auto out_type = real && inverse ? float32 : complex64;

  if (stream.device == Device::gpu) {
    // Perform ND FFT on GPU as a series of 1D FFTs
    if (valid_axes.size() > 1) {
      auto out = in;
      for (int i = valid_axes.size() - 1; i >= 0; i--) {
        // Opposite order for fft vs ifft
        int index = inverse ? valid_axes.size() - i - 1 : i;
        int axis = valid_axes[index];
        // Mirror np.fft.(i)rfftn and perform a real transform
        // only on the final axis.
        bool step_real = (real && index == valid_axes.size() - 1);
        int step_shape = inverse ? out_shape[axis] : in.shape(axis);
        out = fft_impl(out, {step_shape}, {axis}, step_real, inverse, s);
      }
      return out;
    }

    // Guarranteed to be 1D now
    int n_1d = n.back();

    // If n is larger than the maximum size, do a 4 step FFT
    if (n_1d > 2048) {
      return four_step_fft(in, n.back(), valid_axes[0], real, inverse, s);
    }

    // Check if n can be done with the Stockham algorithm
    auto [fast_n, _] = FFT::next_fast_n(n_1d);
    if (fast_n > n_1d) {
      // Precompute twiddle factors in high precision for Bluestein's
      auto [bluestein_n, _] = FFT::next_fast_n(2 * n_1d - 1);
      auto blue_outputs = array::make_arrays(
          {{bluestein_n}, {n_1d}},
          {{complex64, complex64}},
          std::make_shared<BluesteinFFTSetup>(to_stream(Device::cpu), n_1d),
          {});
      array w_q = blue_outputs[0];
      array w_k = blue_outputs[1];
      return array(
          out_shape,
          out_type,
          std::make_shared<FFT>(stream, valid_axes, inverse, real),
          {astype(in, in_type, s), w_q, w_k});
    }
  }

  return array(
      out_shape,
      out_type,
      std::make_shared<FFT>(stream, valid_axes, inverse, real),
      {astype(in, in_type, s)});
}

array fft_impl(
    const array& a,
    const std::vector<int>& axes,
    bool real,
    bool inverse,
    StreamOrDevice s) {
  std::vector<int> n;
  for (auto ax : axes) {
    n.push_back(a.shape(ax));
  }
  if (real && inverse) {
    n.back() = (n.back() - 1) * 2;
  }
  return fft_impl(a, n, axes, real, inverse, s);
}

array fft_impl(const array& a, bool real, bool inverse, StreamOrDevice s) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return fft_impl(a, axes, real, inverse, s);
}

array fftn(
    const array& a,
    const std::vector<int>& n,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, false, false, s);
}
array fftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, false, false, s);
}
array fftn(const array& a, StreamOrDevice s /* = {} */) {
  return fft_impl(a, false, false, s);
}

array ifftn(
    const array& a,
    const std::vector<int>& n,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, false, true, s);
}
array ifftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, false, true, s);
}
array ifftn(const array& a, StreamOrDevice s /* = {} */) {
  return fft_impl(a, false, true, s);
}

array rfftn(
    const array& a,
    const std::vector<int>& n,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, true, false, s);
}
array rfftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, true, false, s);
}
array rfftn(const array& a, StreamOrDevice s /* = {} */) {
  return fft_impl(a, true, false, s);
}

array irfftn(
    const array& a,
    const std::vector<int>& n,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, true, true, s);
}
array irfftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, true, true, s);
}
array irfftn(const array& a, StreamOrDevice s /* = {} */) {
  return fft_impl(a, true, true, s);
}

std::vector<int> prime_factors(int n) {
  int z = 2;
  std::vector<int> factors;
  while (z * z <= n) {
    if (n % z == 0) {
      factors.push_back(z);
      n /= z;
    } else {
      z++;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

// Currently supports:
// All 2 <= N <= 1024
// N < 2^20 when all prime factors of N are < 1024
array four_step_fft(
    const array& a,
    int n,
    int axis,
    bool real,
    bool inverse,
    StreamOrDevice s) {
  // For n that doesn't fit into GPU shared memory, we use the 4 step FFT
  // algorithm.
  auto factors = prime_factors(n);
  int max_factor = *std::max_element(factors.begin(), factors.end());
  // Decompose the problem into two FFTs of size n1 and n2.
  int n1 = 1;
  int n2 = factors.back();
  for (int i = 0; i < factors.size(); i++) {
    int factor = factors[i];
    if (n1 * factor == n) {
      n2 = factor;
      break;
    } else if (n1 * factor > MAX_BLUESTEIN_FFT_SIZE) {
      n2 = std::accumulate(
          factors.begin() + i, factors.end(), 1, std::multiplies<int>());
      break;
    }
    n1 *= factor;
  }
  if (n1 > MAX_BLUESTEIN_FFT_SIZE || n2 > MAX_BLUESTEIN_FFT_SIZE) {
    throw std::runtime_error(
        "Cannot create a plan for N, since prime factors are too large.");
  }
  // (..., n) -> (..., n1, n2)
  std::vector<int> four_step_shape(a.shape());
  auto axis_it = four_step_shape.begin() + axis;
  four_step_shape.erase(axis_it);
  four_step_shape.insert(axis_it, n2);
  four_step_shape.insert(axis_it, n1);

  std::vector<int> twiddle_shape(four_step_shape.size(), 1);
  twiddle_shape[axis] = n1;
  twiddle_shape[axis + 1] = n2;

  array ij =
      expand_dims(arange(n1, s), 1, s) * expand_dims(arange(n2, s), 0, s);
  ij = reshape(ij, twiddle_shape, s);

  array x = reshape(a, four_step_shape, s);
  array twiddles = exp(-2 * M_PI * ij * complex64_t{0.0f, 1.0f} / n, s);
  array step_one = fft_impl(x, {n1}, {axis}, false, false, s) * twiddles;
  array step_two = fft_impl(
      swapaxes(step_one, axis, axis + 1, s), {n2}, {axis}, false, false, s);
  return reshape(step_two, a.shape(), s);
}

} // namespace mlx::core::fft
