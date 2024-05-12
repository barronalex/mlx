// Copyright © 2023 Apple Inc.

#include <numeric>
#include <set>

#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#define MAX_STOCKHAM_FFT_SIZE 4096
#define MAX_BLUESTEIN_FFT_SIZE 2048

namespace mlx::core::fft {

// Forward declarations
array gpu_nd_fft(
    const array& a,
    std::vector<int> n,
    const std::vector<size_t> axes,
    bool real,
    bool inverse,
    std::vector<int> out_shape,
    StreamOrDevice s);

array gpu_irfft(const array& a, int n, int axis, StreamOrDevice s);

array bluestein_fft(
    const array& a,
    int n,
    int axis,
    bool inverse,
    StreamOrDevice s);

array four_step_fft(
    const array& a,
    int n,
    int axis,
    bool inverse,
    StreamOrDevice s);

std::tuple<array, array, array> compute_raders_constants(int n);

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
    // Scatter complex64 is not supported on GPU currently
    in =
        scatter(tmp, std::vector<array>{}, in, std::vector<int>{}, Device::cpu);
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
    // Here are the code paths for GPU FFT:
    // if rader_decomposable(N) && N <= MAX_STOCKHAM:
    //     rader_stockham(N)
    // else if N <= MAX_BLUESTEIN:
    //     fused_bluestein()
    // else if N > MAX_BLUESTEIN && largest_prime_factor(N) <= MAX_BLUESTEIN
    //     four_step()
    // else if N > MAX_BLUESTEIN && larget_prime_factor(N) > MAX_BLUESTEIN:
    //     bluestein()

    if (valid_axes.size() > 1) {
      return gpu_nd_fft(in, n, valid_axes, real, inverse, out_shape, s);
    }

    // Guarranteed to be 1D now
    int n_1d = n.back();
    int axis = valid_axes[0];

    if (n_1d == 1) {
      return astype(a, complex64, s);
    }

    if (out_type == float32) {
      return gpu_irfft(a, n_1d, axis, s);
    }

    FFTPlan plan = FFT::plan_fft(n_1d);
    if (n_1d > MAX_STOCKHAM_FFT_SIZE || plan.bluestein_n > 0 ||
        plan.rader_n > 1) {
      array out = in;
      // If we pass complex input to RFFT, we need to drop the complex part
      // to mirror the CPU implementation.
      array in_complex = astype(astype(in, in_type, s), complex64, s);
      // If n is larger than the maximum bluestein or stockham size, use four
      // step FFT
      if (n_1d > MAX_STOCKHAM_FFT_SIZE ||
          (n_1d > MAX_BLUESTEIN_FFT_SIZE && plan.bluestein_n > 0)) {
        out = four_step_fft(in_complex, n_1d, axis, inverse, s);
      } else if (plan.bluestein_n > 0) {
        out = bluestein_fft(in_complex, n_1d, axis, inverse, s);
      } else {
        // Otherwise, we can use Rader's
        auto [b_q, g_q, g_minus_q] = compute_raders_constants(plan.rader_n);
        out = array(
            in_shape,
            out_type,
            std::make_shared<FFT>(
                stream, valid_axes, inverse, /* real= */ false),
            {in_complex, b_q, g_q, g_minus_q});
      }

      if (in_type == float32) {
        // Efficient RFFT is only implemented for Stockham decomposable n
        // currently
        std::vector<int> starts(in.ndim(), 0);
        std::vector<int> ends(in.shape());
        ends[axis] = n_1d / 2 + 1;
        out = slice(out, starts, ends, s);
      }
      return out;
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

array gpu_nd_fft(
    const array& a,
    std::vector<int> n,
    const std::vector<size_t> axes,
    bool real,
    bool inverse,
    std::vector<int> out_shape,
    StreamOrDevice s) {
  // Perform ND FFT on GPU as a series of 1D FFTs
  auto out = a;
  for (int i = axes.size() - 1; i >= 0; i--) {
    // Opposite order for fft vs ifft
    int index = inverse ? axes.size() - i - 1 : i;
    int axis = axes[index];
    // Mirror np.fft.(i)rfftn and perform a real transform
    // only on the final axis.
    bool step_real = (real && index == axes.size() - 1);
    int step_shape = inverse ? out_shape[axis] : a.shape(axis);
    out = fft_impl(out, {step_shape}, {axis}, step_real, inverse, Device::gpu);
  }
  return out;
}

array gpu_irfft(const array& a, int n, int axis, StreamOrDevice s) {
  std::vector<int> starts(a.ndim(), 0);
  std::vector<int> ends(a.shape());
  std::vector<int> steps(a.ndim(), 1);
  array in = a;
  if (n != 2) {
    starts[axis] = n % 2 == 0 ? -2 : -1;
    ends[axis] = 0;
    steps[axis] = -1;
    array conj = conjugate(slice(a, starts, ends, steps, s), s);
    in = concatenate({a, conj}, axis, s);
  }
  array out = fft_impl(
      in,
      {n},
      {axis},
      /* real= */ false,
      /* inverse= */ true,
      s);
  return astype(out, float32, s);
}

array bluestein_fft(
    const array& a,
    int n,
    int axis,
    bool inverse,
    StreamOrDevice s) {
  int bluestein_n = FFT::next_fast_n(2 * n - 1);
  // Precompute twiddle factors in high precision for Bluestein's
  auto blue_outputs = array::make_arrays(
      {{bluestein_n}, {n}},
      {{complex64, complex64}},
      std::make_shared<BluesteinFFTSetup>(to_stream(Device::cpu), n),
      {});
  // If N is small enough, use the fused implementation
  array w_q = blue_outputs[0];
  array w_k = blue_outputs[1];
  std::vector<size_t> axes;
  axes.push_back(axis);
  if (bluestein_n <= MAX_STOCKHAM_FFT_SIZE) {
    return array(
        a.shape(),
        complex64,
        std::make_shared<FFT>(to_stream(s), axes, inverse, false),
        {a, w_q, w_k});
  } else {
    // Broadcast w_k and w_q to the appropriate shapes
    std::vector<int> broadcast_shape(a.ndim(), 1);
    broadcast_shape[axis] = a.shape(axis);
    w_k = reshape(w_k, broadcast_shape);
    broadcast_shape[axis] = bluestein_n;
    w_q = reshape(w_q, broadcast_shape);

    // Pad out to the bluestein n size
    std::vector<std::pair<int, int>> pads;
    for (int i = 0; i < a.ndim(); i++) {
      if (axis == i) {
        pads.push_back({0, bluestein_n - n});
      } else {
        pads.push_back({0, 0});
      }
    }

    // Do the convolution via FFT
    array out = pad(a * w_k, pads, array(0), s);
    out = fft_impl(
        out, {bluestein_n}, {axis}, /* real= */ false, /* inverse= */ false, s);
    out = fft_impl(
        out * w_q,
        {bluestein_n},
        {axis},
        /* real= */ false,
        /* inverse= */ true,
        s);
    int offset = bluestein_n - (2 * n - 1);

    // Slice back into the original FFT size
    std::vector<int> starts(a.ndim(), 0);
    std::vector<int> ends(a.shape());
    starts[axis] = -offset - n;
    ends[axis] = -offset;
    out = slice(out, starts, ends, s);

    out = w_k * out;
    return out;
  }
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

// For n that doesn't fit into GPU shared memory, we use the 4 step FFT
// algorithm.
array four_step_fft(
    const array& a,
    int n,
    int axis,
    bool inverse,
    StreamOrDevice s) {
  array in = inverse ? conjugate(a) : a;

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
    // Prime factors are too large for the fused bluestein implementation
    // so we fallback to doing a manual bluestein.
    array out = bluestein_fft(in, n, axis, false, s);
    return inverse ? conjugate(out) / n : out;
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

  array x = reshape(in, four_step_shape, s);
  array twiddles = exp(-2 * M_PI * ij * complex64_t{0.0f, 1.0f} / n, s);
  array step_one =
      fft_impl(x, {n1}, {axis}, /* real= */ false, /* inverse= */ false, s) *
      twiddles;
  array step_two = fft_impl(
      swapaxes(step_one, axis, axis + 1, s),
      {n2},
      {axis},
      /* real= */ false,
      /* inverse= */ false,
      s);
  array out = reshape(step_two, a.shape(), s);
  out = inverse ? conjugate(out) / n : out;
  return out;
}

std::tuple<array, array, array> compute_raders_constants(int raders_n) {
  int proot = primitive_root(raders_n);
  // Fermat's little theorem
  int inv = mod_exp(proot, raders_n - 2, raders_n);
  // Now get the g_minus_q sequence
  std::vector<short> g_q(raders_n - 1);
  std::vector<short> g_minus_q(raders_n - 1);
  for (int i = 0; i < raders_n - 1; i++) {
    g_q[i] = mod_exp(proot, i, raders_n);
    g_minus_q[i] = mod_exp(inv, i, raders_n);
  }
  array g_q_arr(g_q.begin(), {raders_n - 1});
  array g_minus_q_arr(g_minus_q.begin(), {raders_n - 1});
  array b_q =
      exp(complex64_t{0.0f, 2.0f} * astype(g_minus_q_arr, float32) / raders_n *
          -M_PI);
  array b_q_fft = fft_impl(
      b_q,
      {raders_n - 1},
      {0},
      /* real= */ false,
      /* inverse= */ false,
      Device::cpu);
  return std::make_tuple(b_q_fft, g_q_arr, g_minus_q_arr);
}

} // namespace mlx::core::fft
