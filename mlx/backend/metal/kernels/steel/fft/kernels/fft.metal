// Copyright © 2024 Apple Inc.

// Metal FFT using Stockham's algorithm
//
// References:
// - VkFFT (https://github.com/DTolm/VkFFT)
// - Eric Bainville's excellent page (http://www.bealto.com/gpu-fft.html)

#include <metal_common>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/fft/radix.h"
#include "mlx/backend/metal/kernels/steel/fft/readwrite.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Plan:
// - RFFT
// - Strided reads

#define MAX_RADIX 13
// Reached when elems_per_thread_ = 8, max_radix = 11/13
// and some threads have to do 2 radix 8s requiring 16 float2s.
#define MAX_OUTPUT_SIZE 16

// Specialize for a particular value of N at runtime
STEEL_CONST bool inv_ [[function_constant(0)]];
STEEL_CONST bool is_power_of_2_ [[function_constant(1)]];
STEEL_CONST int elems_per_thread_ [[function_constant(2)]];
// rader_m = n / rader_n
STEEL_CONST int rader_m_ [[function_constant(3)]];
// Stockham steps
STEEL_CONST int radix_13_steps_ [[function_constant(4)]];
STEEL_CONST int radix_11_steps_ [[function_constant(5)]];
STEEL_CONST int radix_8_steps_ [[function_constant(6)]];
STEEL_CONST int radix_7_steps_ [[function_constant(7)]];
STEEL_CONST int radix_6_steps_ [[function_constant(8)]];
STEEL_CONST int radix_5_steps_ [[function_constant(9)]];
STEEL_CONST int radix_4_steps_ [[function_constant(10)]];
STEEL_CONST int radix_3_steps_ [[function_constant(11)]];
STEEL_CONST int radix_2_steps_ [[function_constant(12)]];
// Rader steps
STEEL_CONST int rader_13_steps_ [[function_constant(13)]];
STEEL_CONST int rader_11_steps_ [[function_constant(14)]];
STEEL_CONST int rader_8_steps_ [[function_constant(15)]];
STEEL_CONST int rader_7_steps_ [[function_constant(16)]];
STEEL_CONST int rader_6_steps_ [[function_constant(17)]];
STEEL_CONST int rader_5_steps_ [[function_constant(18)]];
STEEL_CONST int rader_4_steps_ [[function_constant(19)]];
STEEL_CONST int rader_3_steps_ [[function_constant(20)]];
STEEL_CONST int rader_2_steps_ [[function_constant(21)]];

METAL_FUNC float2 get_twiddle(int k, int p) {
  float theta = -2.0f * k * M_PI_F / p;

  float2 twiddle = {metal::fast::cos(theta), metal::fast::sin(theta)};
  return twiddle;
}

// See "radix.h" for radix codelets
typedef void (*RadixFunc)(thread float2*, thread float2*);

// Perform a single radix n butterfly with appropriate twiddles
template <int radix, RadixFunc radix_func>
METAL_FUNC void radix_n(
    int i,
    int p,
    thread float2* x,
    thread short* indices,
    thread float2* y) {
  // i: the index in the overall DFT that we're processing.
  // p: the size of the DFTs we're merging at this step.
  // m: how many threads are working on this DFT.
  int k, j;

  // Use faster bitwise operations when working with powers of two
  constexpr bool radix_p_2 = (radix & (radix - 1)) == 0;
  if (radix_p_2 && is_power_of_2_) {
    constexpr short power = __builtin_ctz(radix);
    k = i & (p - 1);
    j = ((i - k) << power) + k;
  } else {
    k = i % p;
    j = (i / p) * radix * p + k;
  }

  // Apply twiddles based on where in the decomposition we are
  if (p > 1) {
    float2 twiddle_1 = get_twiddle(k, radix * p);
    float2 twiddle = twiddle_1;
    x[1] = complex_mul(x[1], twiddle);

    STEEL_PRAGMA_UNROLL
    for (int t = 2; t < radix; t++) {
      twiddle = complex_mul(twiddle, twiddle_1);
      x[t] = complex_mul(x[t], twiddle);
    }
  }

  radix_func(x, y);

  STEEL_PRAGMA_UNROLL
  for (int t = 0; t < radix; t++) {
    indices[t] = j + t * p;
  }
}

// Perform all the radix steps required for a
// particular radix size n.
template <int radix, RadixFunc radix_func>
METAL_FUNC void radix_n_steps(
    int i,
    thread int* p,
    int m,
    int n,
    int num_steps,
    thread float2* inputs,
    thread short* indices,
    thread float2* values,
    threadgroup float2* buf) {
  int m_r = n / radix;
  // When combining different sized radices, we have to do
  // multiple butterflies in a single thread.
  // E.g. n = 28 = 4 * 7
  // 4 threads, 7 elems_per_thread
  // All threads do 1 radix7 butterfly.
  // 3 threads do 2 radix4 butterflies.
  // 1 thread does 1 radix4 butterfly.
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;

  int index = 0;
  int r_index = 0;
  for (int s = 0; s < num_steps; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          inputs[r] = buf[index + r * m_r];
        }
        radix_n<radix, radix_func>(
            index, *p, inputs, indices + t * radix, values + t * radix);
      }
    }

    // Wait until all threads have read their inputs into thread local mem
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < max_radices_per_thread; t++) {
      index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          r_index = t * radix + r;
          buf[indices[r_index]] = values[r_index];
        }
      }
    }

    // Wait until all threads have written back to threadgroup mem
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *p *= radix;
  }
}

#define RADIX_STEP(radix, radix_func, num_steps) \
  radix_n_steps<radix, radix_func>(              \
      fft_idx, p, m, n, num_steps, inputs, indices, values, buf);

METAL_FUNC void
radix_fft(int fft_idx, thread int* p, int m, int n, threadgroup float2* buf) {
  float2 inputs[MAX_RADIX];
  short indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

  RADIX_STEP(2, radix2, radix_2_steps_);
  RADIX_STEP(3, radix3, radix_3_steps_);
  RADIX_STEP(4, radix4, radix_4_steps_);
  RADIX_STEP(5, radix5, radix_5_steps_);
  RADIX_STEP(6, radix6, radix_6_steps_);
  RADIX_STEP(7, radix7, radix_7_steps_);
  RADIX_STEP(8, radix8, radix_8_steps_);
  RADIX_STEP(11, radix11, radix_11_steps_);
  RADIX_STEP(13, radix13, radix_13_steps_);
}

METAL_FUNC void
rader_fft(int fft_idx, thread int* p, int m, int n, threadgroup float2* buf) {
  float2 inputs[MAX_RADIX];
  short indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

  RADIX_STEP(2, radix2, rader_2_steps_);
  RADIX_STEP(3, radix3, rader_3_steps_);
  RADIX_STEP(4, radix4, rader_4_steps_);
  RADIX_STEP(5, radix5, rader_5_steps_);
  RADIX_STEP(6, radix6, rader_6_steps_);
  RADIX_STEP(7, radix7, rader_7_steps_);
  RADIX_STEP(8, radix8, rader_8_steps_);
  RADIX_STEP(11, radix11, rader_11_steps_);
  RADIX_STEP(13, radix13, rader_13_steps_);
}

// Each FFT is computed entirely in shared GPU memory.
//
// N is decomposed into radix-n DFTs:
// e.g. 128 = 2 * 4 * 4 * 4
template <int tg_mem_size, typename in_T>
[[kernel]] void fft(
    const device in_T* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  threadgroup float2 shared_in[tg_mem_size];

  thread ReadWriter<in_T> read_writer = ReadWriter<in_T>(
      in, &shared_in[0], out, n, batch_size, elems_per_thread_, elem, grid);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load();

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  int i = elem.z; // Thread index in DFT
  int m = grid.z; // Threads per DFT
  int tg_idx = elem.y * n; // Index of this DFT in threadgroup
  threadgroup float2* buf = &shared_in[tg_idx];

  // Do the FFT
  radix_fft(i, &p, m, n, buf);

  read_writer.write();
}

template <int tg_mem_size>
[[kernel]] void rader_fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    const device float2* raders_b_q [[buffer(2)]],
    const device short* raders_g_q [[buffer(3)]],
    const device short* raders_g_minus_q [[buffer(4)]],
    constant const int& n,
    constant const int& batch_size,
    constant const int& rader_n,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // Use Rader's algorithm to compute fast FFTs
  // when a prime factor `p` of `n` is greater than 13 but
  // has `p - 1` Stockham decomposable into to prime factors <= 13.
  //
  // E.g. n = 102
  //        = 2 * 3 * 17
  // .      = 2 * 3 * RADER(16)
  // .      = 2 * 3 * RADER(4 * 4)
  //
  // In numpy:
  //   x_perm = x[g_q]
  //   y = np.fft.fft(x_perm) * b_q
  //   z = np.fft.ifft(y) + x[0]
  //   out = z[g_minus_q]
  //   out[0]  = x[1:].sum()
  //
  // Where the g_q and g_minus_q are permutations formed
  // by the group under multiplicative modulo N using the
  // primitive root of N and b_q is a constant.
  // See https://en.wikipedia.org/wiki/Rader%27s_FFT_algorithm
  //
  // Rader's uses fewer operations than Bluestein's and so
  // is more accurate. It's also faster in most cases.
  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  int batch_idx = elem.x * grid.y * n + tg_idx;

  // The number of the threads we're using for each DFT
  int m = grid.z;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* buf = &shared_in[tg_idx];
  // Account for possible extra threadgroups
  int grid_index = elem.x * grid.y + elem.y;
  if (grid_index >= batch_size) {
    return;
  }

  // rader_m = n / rader_n;
  int rader_m = rader_m_;

  int max_index = n - rader_m - 1;

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short g_q = raders_g_q[index / rader_m];
    buf[index + rader_m] =
        in[batch_idx + rader_m + (g_q - 1) * rader_m + index % rader_m];
    if (inv_) {
      buf[index + rader_m].y = -buf[index + rader_m].y;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  // Rader FFT on x[rader_m:]
  rader_fft(fft_idx, &p, m, n - rader_m, buf + rader_m);

  // x_1 + ... + x_n is computed for us in the first FFT step so
  // we save it in the first rader_m indices of the array for later.
  int x_sum_index = metal::min(fft_idx, rader_m - 1);
  buf[x_sum_index] = buf[rader_m + x_sum_index * (rader_n - 1)];

  float2 temp[MAX_RADIX];
  float2 inv = {1.0f, -1.0f};
  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short interleaved_index =
        index / rader_m + (index % rader_m) * (rader_n - 1);
    temp[e] = complex_mul(
        buf[rader_m + interleaved_index],
        raders_b_q[interleaved_index % (rader_n - 1)]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    buf[rader_m + index] = temp[e] * inv;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  p = 1;
  // Rader IFFT on x[rader_m:]
  rader_fft(fft_idx, &p, m, n - rader_m, buf + rader_m);

  short x_0_index =
      metal::min(fft_idx * elems_per_thread_ / (rader_n - 1), rader_m - 1);
  // We have to load two x_0s for each thread since sometimes
  // elems_per_thread_ crosses a boundary.
  // E.g. with n = 34, rader_n = 17, elems_per_thread_ = 4
  // 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 8 8
  // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  float2 x_0[2] = {in[batch_idx + x_0_index], in[batch_idx + x_0_index + 1]};
  if (inv_) {
    x_0[0].y = -x_0[0].y;
    x_0[1].y = -x_0[1].y;
  }

  float2 rader_inv_factor = {1.0f / (rader_n - 1), -1.0f / (rader_n - 1)};

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, n - rader_m - 1);
    short diff_index = index / (rader_n - 1) - x_0_index;
    temp[e] = buf[rader_m + index] * rader_inv_factor + x_0[diff_index];
  }

  // Use the sum of elements that was computed in the first FFT
  float2 x_sum = buf[x_0_index] + x_0[0];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short g_q_index = index % (rader_n - 1);
    short g_q = raders_g_minus_q[g_q_index];
    short out_index = index - g_q_index + g_q + (index / (rader_n - 1));
    buf[out_index] = temp[e];
  }

  buf[x_0_index * rader_n] = x_sum;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  p = rader_n;
  radix_fft(fft_idx, &p, m, n, buf);

  // Write to device
  float2 inv_factor = {1.0f / n, -1.0f / n};
  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, n - 1);
    float2 elem = buf[index];
    if (inv_) {
      elem *= inv_factor;
    }
    out[batch_idx + index] = elem;
  }
}

template <int tg_mem_size>
[[kernel]] void bluestein_fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    const device float2* w_q [[buffer(2)]],
    const device float2* w_k [[buffer(3)]],
    constant const int& length,
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // Computes arbitrary length FFTs with Bluestein's algorithm
  //
  // In numpy:
  //   bluestein_n = next_power_of_2(2*n - 1)
  //   out = w_k * np.fft.ifft(np.fft.fft(w_k * in, bluestein_n) * w_q)
  //
  // Where w_k and w_q are precomputed on CPU in high precision as:
  //   w_k = np.exp(-1j * np.pi / n * (np.arange(-n + 1, n) ** 2))
  //   w_q = np.fft.fft(1/w_k[-n:])
  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  int batch_idx = elem.x * grid.y * length + elem.y * length;

  int m = grid.z;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* buf = &shared_in[tg_idx];

  // Account for possible extra threadgroups
  int grid_index = elem.x * grid.y + elem.y;
  if (grid_index >= batch_size) {
    return;
  }

  // load input into shared memory
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    if (index < length) {
      float2 elem = in[batch_idx + index];
      if (inv_) {
        elem.y = -elem.y;
      }
      buf[index] = complex_mul(elem, w_k[index]);
    } else {
      buf[index] = 0.0;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  radix_fft(fft_idx, &p, m, n, buf);

  float2 inv = float2(1.0f, -1.0f);
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    buf[index] = complex_mul(buf[index], w_q[index]) * inv;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ifft
  p = 1;
  radix_fft(fft_idx, &p, m, n, buf);

  float2 inv_factor = {1.0f / n, -1.0f / n};
  float2 inv_factor_overall = {1.0f / length, -1.0f / length};

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    if (index < length) {
      float2 elem = buf[index + length - 1] * inv_factor;
      elem = complex_mul(elem, w_k[index]);
      if (inv_) {
        elem *= inv_factor_overall;
      }
      out[batch_idx + index] = elem;
    }
  }
}

template <int tg_mem_size>
[[kernel]] void rfft(
    const device float* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // For RFFT, we interleave batches of two real sequences into one complex one:
  //
  // z_k = x_k + j.y_k
  // X_k = (Z_k + Z_(N-k)*) / 2
  // Y_k = -j * ((Z_k - Z_(N-k)*) / 2)
  //
  // This roughly doubles the throughput over the regular FFT.
  int n_over_2 = (n / 2) + 1;

  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  int batch_idx = elem.x * grid.y * 2 * n + elem.y * 2 * n;
  int batch_idx_out = elem.x * grid.y * 2 * n_over_2 + elem.y * 2 * n_over_2;

  int m = grid.z;

  // Account for possible extra threadgroups
  int grid_index = elem.x * grid.y + elem.y;
  if (grid_index * 2 >= batch_size) {
    return;
  }

  int next_in = n;
  int next_out = n_over_2;
  // No out of bounds accesses on odd batch sizes
  if (batch_size % 2 == 1 && grid_index * 2 == batch_size - 1) {
    next_in = 0;
    next_out = 0;
  }

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* buf = &shared_in[tg_idx];

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = metal::min(fft_idx + t * m, n - 1);
    buf[index].x = in[batch_idx + index];
    buf[index].y = in[batch_idx + index + next_in];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  radix_fft(fft_idx, &p, m, n, buf);

  float2 conj = {1, -1};
  float2 minus_j = {0, -1};
  for (int t = 0; t < elems_per_thread_ / 2; t++) {
    int index = metal::min(fft_idx + t * m, n_over_2 - 1);
    // x_0 = z_0.real
    // y_0 = z_0.imag
    if (index == 0) {
      out[batch_idx_out + index] = {buf[index].x, 0};
      out[batch_idx_out + index + next_out] = {buf[index].y, 0};
    } else {
      float2 x_k = buf[index];
      float2 x_n_minus_k = buf[n - index] * conj;
      out[batch_idx_out + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx_out + index + next_out] =
          complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
  // Add in elements up to n/2 + 1
  int num_left = n_over_2 - (elems_per_thread_ / 2 * m);
  if (fft_idx < num_left) {
    int index = metal::min(fft_idx + elems_per_thread_ / 2 * m, n_over_2 - 1);
    float2 x_k = buf[index];
    float2 x_n_minus_k = buf[n - index] * conj;
    out[batch_idx_out + index] = (x_k + x_n_minus_k) / 2;
    out[batch_idx_out + index + next_out] =
        complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
  }
}

template <int tg_mem_size>
[[kernel]] void four_step_fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant const int& n1,
    constant const int& n2,
    constant const int& batch_size,
    constant const bool& first_step,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // Fast four step FFT implementation for powers of 2.
  int n = first_step ? n1 : n2;
  int stride = first_step ? n2 : n1;

  int fft_idx = elem.z;
  int batch_idx = elem.x * grid.y * n + elem.y * n;

  // The number of the threads we're using for each DFT
  int m = grid.z;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* buf = &shared_in[elem.y * n];

  // Account for possible extra threadgroups
  int grid_index = elem.x * grid.y + elem.y;
  if (grid_index >= batch_size) {
    return;
  }

  int overall_n = n1 * n2;

  // For strided reads/writes we use the threadgroup batch dimension
  // to ensure consecutive memory accesses. e.g. strided read:
  //
  // device   | shared mem
  // 0 1 2 3  |  0 - - -
  // - - - -  |  1 - - -
  // - - - -  |  2 - - -
  // - - - -  |  3 - - -
  int coalesce_width = grid.y;
  int tg_idx = elem.y * grid.z + elem.z;
  int shared_idx = (tg_idx % coalesce_width) * n +
      tg_idx / coalesce_width * elems_per_thread_;
  int outer_batch_size = (stride / coalesce_width);
  int base_batch_idx = (elem.x % outer_batch_size) * coalesce_width +
      overall_n * (elem.x / outer_batch_size);
  int device_idx = base_batch_idx +
      tg_idx / coalesce_width * elems_per_thread_ * stride +
      tg_idx % coalesce_width;

  if (first_step) {
    for (int e = 0; e < elems_per_thread_; e++) {
      shared_in[shared_idx + e] = in[device_idx + e * stride];
      if (inv_) {
        shared_in[shared_idx + e].y = -shared_in[shared_idx + e].y;
      }
    }
  } else {
    for (int e = 0; e < elems_per_thread_; e++) {
      int index = metal::min(fft_idx + e * m, n - 1);
      buf[index] = in[batch_idx + index];
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  radix_fft(fft_idx, &p, m, n, buf);

  if (first_step) {
    for (int e = 0; e < elems_per_thread_; e++) {
      int combined_idx = (device_idx + e * stride) % overall_n;
      int ij = (combined_idx / stride) * (combined_idx % stride);
      // Apply four step twiddles after first step
      float2 twiddle = get_twiddle(ij, overall_n);
      out[device_idx + e * stride] =
          complex_mul(shared_in[shared_idx + e], twiddle);
    }
  } else {
    for (int e = 0; e < elems_per_thread_; e++) {
      float2 output = shared_in[shared_idx + e];
      if (inv_) {
        output *= float2(1.0f / overall_n, -1.0f / overall_n);
      }
      out[device_idx + e * stride] = output;
    }
  }
}

#define instantiate_fft(tg_mem_size, in_T)                                  \
  template [[host_name("fft_mem_" #tg_mem_size "_" #in_T)]] [[kernel]] void \
  fft<tg_mem_size, in_T>(                                                   \
      const device in_T* in [[buffer(0)]],                                  \
      device float2* out [[buffer(1)]],                                     \
      constant const int& n,                                                \
      constant const int& batch_size,                                       \
      uint3 elem [[thread_position_in_grid]],                               \
      uint3 grid [[threads_per_grid]]);

#define instantiate_rader(tg_mem_size)                                  \
  template [[host_name("rader_fft_mem_" #tg_mem_size)]] [[kernel]] void \
  rader_fft<tg_mem_size>(                                               \
      const device float2* in [[buffer(0)]],                            \
      device float2* out [[buffer(1)]],                                 \
      const device float2* raders_b_q [[buffer(2)]],                    \
      const device short* raders_g_q [[buffer(3)]],                     \
      const device short* raders_g_minus_q [[buffer(4)]],               \
      constant const int& n,                                            \
      constant const int& batch_size,                                   \
      constant const int& rader_n,                                      \
      uint3 elem [[thread_position_in_grid]],                           \
      uint3 grid [[threads_per_grid]]);

#define instantiate_bluestein(tg_mem_size)                                  \
  template [[host_name("bluestein_fft_mem_" #tg_mem_size)]] [[kernel]] void \
  bluestein_fft<tg_mem_size>(                                               \
      const device float2* in [[buffer(0)]],                                \
      device float2* out [[buffer(1)]],                                     \
      const device float2* w_q [[buffer(2)]],                               \
      const device float2* w_k [[buffer(2)]],                               \
      constant const int& length,                                           \
      constant const int& n,                                                \
      constant const int& batch_size,                                       \
      uint3 elem [[thread_position_in_grid]],                               \
      uint3 grid [[threads_per_grid]]);

#define instantiate_rfft(tg_mem_size)                              \
  template [[host_name("rfft_mem_" #tg_mem_size)]] [[kernel]] void \
  rfft<tg_mem_size>(                                               \
      const device float* in [[buffer(0)]],                        \
      device float2* out [[buffer(1)]],                            \
      constant const int& n,                                       \
      constant const int& batch_size,                              \
      uint3 elem [[thread_position_in_grid]],                      \
      uint3 grid [[threads_per_grid]]);

#define instantiate_four_step(tg_mem_size)                              \
  template [[host_name("four_step_mem_" #tg_mem_size)]] [[kernel]] void \
  four_step_fft<tg_mem_size>(                                           \
      const device float2* in [[buffer(0)]],                            \
      device float2* out [[buffer(1)]],                                 \
      constant const int& n1,                                           \
      constant const int& n2,                                           \
      constant const int& batch_size,                                   \
      constant const bool& first_step,                                  \
      uint3 elem [[thread_position_in_grid]],                           \
      uint3 grid [[threads_per_grid]]);

// clang-format off
#define instantiate_ffts(tg_mem_size)                        \
  instantiate_fft(tg_mem_size, float2) \
  instantiate_fft(tg_mem_size, float) \
  instantiate_rader(tg_mem_size) \
  instantiate_bluestein(tg_mem_size) \
  instantiate_four_step(tg_mem_size)

// It's substantially faster to statically define the
// threadgroup memory size rather than using
// `setThreadgroupMemoryLength` on the compute encoder.
// For non-power of 2 sizes we round up the shared memory.
instantiate_ffts(256)
instantiate_ffts(512)
instantiate_ffts(1024)
instantiate_ffts(2048)
// 4096 is the max that will fit into 32KB of threadgroup memory.
instantiate_ffts(4096) // clang-format on