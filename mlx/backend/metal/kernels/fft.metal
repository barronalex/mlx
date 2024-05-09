// Copyright © 2024 Apple Inc.

// Metal FFT using Stockham's algorithm
//
// References:
// - VkFFT (https://github.com/DTolm/VkFFT)
// - Eric Bainville's excellent page (http://www.bealto.com/gpu-fft.html)

#include <metal_common>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define MAX_OUTPUT_SIZE 16
#define MAX_ELEMS_PER_THREAD 8
#define MAX_RADIX 13

// Specialize for a particular value of N at runtime
constant bool inv_ [[function_constant(0)]];
constant bool is_power_of_2_ [[function_constant(1)]];
constant int elems_per_thread_ [[function_constant(2)]];
// Signal which radices we need to read/write to global memory on
constant bool is_rader_ [[function_constant(3)]];
// Stockham steps
constant int radix_13_steps_ [[function_constant(4)]];
constant int radix_11_steps_ [[function_constant(5)]];
constant int radix_8_steps_ [[function_constant(6)]];
constant int radix_7_steps_ [[function_constant(7)]];
constant int radix_5_steps_ [[function_constant(8)]];
constant int radix_4_steps_ [[function_constant(9)]];
constant int radix_3_steps_ [[function_constant(10)]];
constant int radix_2_steps_ [[function_constant(11)]];
// Rader steps
constant int rader_13_steps_ [[function_constant(12)]];
constant int rader_11_steps_ [[function_constant(13)]];
constant int rader_8_steps_ [[function_constant(14)]];
constant int rader_7_steps_ [[function_constant(15)]];
constant int rader_5_steps_ [[function_constant(16)]];
constant int rader_4_steps_ [[function_constant(17)]];
constant int rader_3_steps_ [[function_constant(18)]];
constant int rader_2_steps_ [[function_constant(19)]];
constant int rader_m_ [[function_constant(20)]];

METAL_FUNC float2 complex_mul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex mul followed by conjugate
METAL_FUNC float2 complex_mul_conj(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, -a.x * b.y - a.y * b.x);
}

METAL_FUNC float2 get_twiddle(int k, int p) {
  float theta = -2.0f * k * M_PI_F / p;

  float2 twiddle = {metal::fast::cos(theta), metal::fast::sin(theta)};
  return twiddle;
}

typedef void (*RadixFunc)(thread float2*, thread float2*);

// Radix kernels
//
// We provide optimized, single threaded Radix codelets
// for n=2,3,4,5,6,7,8,10,11,12,13.
//
// For n=2,3,4,5,6 we hand write the codelets.
// For n=8,10,12 we combine smaller codelets.
// For n=7,11,13 we use Rader's algorithm which decomposes
// them into (n-1)=6,10,12 codelets.

METAL_FUNC void radix2(thread float2* x, thread float2* y) {
  y[0] = x[0] + x[1];
  y[1] = x[0] - x[1];
}

METAL_FUNC void radix3(thread float2* x, thread float2* y) {
  float pi_2_3 = -0.8660254037844387;

  float2 a_1 = x[1] + x[2];
  float2 a_2 = x[1] - x[2];

  y[0] = x[0] + a_1;
  float2 b_1 = x[0] - 0.5 * a_1;
  float2 b_2 = pi_2_3 * a_2;

  float2 b_2_j = {-b_2.y, b_2.x};
  y[1] = b_1 + b_2_j;
  y[2] = b_1 - b_2_j;
}

METAL_FUNC void radix4(thread float2* x, thread float2* y) {
  float2 z_0 = x[0] + x[2];
  float2 z_1 = x[0] - x[2];
  float2 z_2 = x[1] + x[3];
  float2 z_3 = x[1] - x[3];
  float2 z_3_i = {z_3.y, -z_3.x};

  y[0] = z_0 + z_2;
  y[1] = z_1 + z_3_i;
  y[2] = z_0 - z_2;
  y[3] = z_1 - z_3_i;
}

METAL_FUNC void radix5(thread float2* x, thread float2* y) {
  float2 root_5_4 = 0.5590169943749475;
  float2 sin_2pi_5 = 0.9510565162951535;
  float2 sin_1pi_5 = 0.5877852522924731;

  float2 a_1 = x[1] + x[4];
  float2 a_2 = x[2] + x[3];
  float2 a_3 = x[1] - x[4];
  float2 a_4 = x[2] - x[3];

  float2 a_5 = a_1 + a_2;
  float2 a_6 = root_5_4 * (a_1 - a_2);
  float2 a_7 = x[0] - a_5 / 4;
  float2 a_8 = a_7 + a_6;
  float2 a_9 = a_7 - a_6;
  float2 a_10 = sin_2pi_5 * a_3 + sin_1pi_5 * a_4;
  float2 a_11 = sin_1pi_5 * a_3 - sin_2pi_5 * a_4;
  float2 a_10_j = {a_10.y, -a_10.x};
  float2 a_11_j = {a_11.y, -a_11.x};

  y[0] = x[0] + a_5;
  y[1] = a_8 + a_10_j;
  y[2] = a_9 + a_11_j;
  y[3] = a_9 - a_11_j;
  y[4] = a_8 - a_10_j;
}

METAL_FUNC void radix6(thread float2* x, thread float2* y) {
  float sin_pi_3 = 0.8660254037844387;
  float2 a_1 = x[2] + x[4];
  float2 a_2 = x[0] - a_1 / 2;
  float2 a_3 = sin_pi_3 * (x[2] - x[4]);
  float2 a_4 = x[5] + x[1];
  float2 a_5 = x[3] - a_4 / 2;
  float2 a_6 = sin_pi_3 * (x[5] - x[1]);
  float2 a_7 = x[0] + a_1;

  float2 a_3_i = {a_3.y, -a_3.x};
  float2 a_6_i = {a_6.y, -a_6.x};
  float2 a_8 = a_2 + a_3_i;
  float2 a_9 = a_2 - a_3_i;
  float2 a_10 = x[3] + a_4;
  float2 a_11 = a_5 + a_6_i;
  float2 a_12 = a_5 - a_6_i;

  y[0] = a_7 + a_10;
  y[1] = a_8 - a_11;
  y[2] = a_9 + a_12;
  y[3] = a_7 - a_10;
  y[4] = a_8 + a_11;
  y[5] = a_9 - a_12;
}

METAL_FUNC void radix7(thread float2* x, thread float2* y) {
  // Rader's algorithm
  float2 inv = {1 / 6.0, -1 / 6.0};

  // fft
  float2 in1[6] = {x[1], x[3], x[2], x[6], x[4], x[5]};
  radix6(in1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(2.44013336, -1.02261879));
  y[3] = complex_mul_conj(y[3], float2(2.37046941, -1.17510629));
  y[4] = complex_mul_conj(y[4], float2(0, -2.64575131));
  y[5] = complex_mul_conj(y[5], float2(2.37046941, 1.17510629));
  y[6] = complex_mul_conj(y[6], float2(-2.44013336, -1.02261879));

  // ifft
  radix6(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[5] = x[2] * inv + x[0];
  y[4] = x[3] * inv + x[0];
  y[6] = x[4] * inv + x[0];
  y[2] = x[5] * inv + x[0];
  y[3] = x[6] * inv + x[0];
}

METAL_FUNC void radix8(thread float2* x, thread float2* y) {
  float cos_pi_4 = 0.7071067811865476;
  float2 w_0 = {cos_pi_4, -cos_pi_4};
  float2 w_1 = {-cos_pi_4, -cos_pi_4};
  float2 temp[8] = {x[0], x[2], x[4], x[6], x[1], x[3], x[5], x[7]};
  radix4(temp, x);
  radix4(temp + 4, x + 4);

  y[0] = x[0] + x[4];
  y[4] = x[0] - x[4];
  float2 x_5 = complex_mul(x[5], w_0);
  y[1] = x[1] + x_5;
  y[5] = x[1] - x_5;
  float2 x_6 = {x[6].y, -x[6].x};
  y[2] = x[2] + x_6;
  y[6] = x[2] - x_6;
  float2 x_7 = complex_mul(x[7], w_1);
  y[3] = x[3] + x_7;
  y[7] = x[3] - x_7;
}

template <bool raders_perm>
METAL_FUNC void radix10(thread float2* x, thread float2* y) {
  float2 w[4];
  w[0] = {0.8090169943749475, -0.5877852522924731};
  w[1] = {0.30901699437494745, -0.9510565162951535};
  w[2] = {-w[1].x, w[1].y};
  w[3] = {-w[0].x, w[0].y};

  if (raders_perm) {
    float2 temp[10] = {
        x[0], x[3], x[4], x[8], x[2], x[1], x[7], x[9], x[6], x[5]};
    radix5(temp, x);
    radix5(temp + 5, x + 5);
  } else {
    float2 temp[10] = {
        x[0], x[2], x[4], x[6], x[8], x[1], x[3], x[5], x[7], x[9]};
    radix5(temp, x);
    radix5(temp + 5, x + 5);
  }

  y[0] = x[0] + x[5];
  y[5] = x[0] - x[5];
  for (int t = 1; t < 5; t++) {
    float2 a = complex_mul(x[t + 5], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 5] = x[t] - a;
  }
}

METAL_FUNC void radix11(thread float2* x, thread float2* y) {
  // Raders Algorithm
  float2 inv = {1 / 10.0, -1 / 10.0};

  // fft
  radix10<true>(x + 1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(0.955301878, -3.17606649));
  y[3] = complex_mul_conj(y[3], float2(2.63610556, 2.01269656));
  y[4] = complex_mul_conj(y[4], float2(2.54127802, 2.13117479));
  y[5] = complex_mul_conj(y[5], float2(2.07016210, 2.59122150));
  y[6] = complex_mul_conj(y[6], float2(0, -3.31662479));
  y[7] = complex_mul_conj(y[7], float2(2.07016210, -2.59122150));
  y[8] = complex_mul_conj(y[8], float2(-2.54127802, 2.13117479));
  y[9] = complex_mul_conj(y[9], float2(2.63610556, -2.01269656));
  y[10] = complex_mul_conj(y[10], float2(-0.955301878, -3.17606649));

  // ifft
  radix10<false>(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[6] = x[2] * inv + x[0];
  y[3] = x[3] * inv + x[0];
  y[7] = x[4] * inv + x[0];
  y[9] = x[5] * inv + x[0];
  y[10] = x[6] * inv + x[0];
  y[5] = x[7] * inv + x[0];
  y[8] = x[8] * inv + x[0];
  y[4] = x[9] * inv + x[0];
  y[2] = x[10] * inv + x[0];
}

template <bool raders_perm>
METAL_FUNC void radix12(thread float2* x, thread float2* y) {
  float2 w[6];
  float sin_pi_3 = 0.8660254037844387;
  w[0] = {sin_pi_3, -0.5};
  w[1] = {0.5, -sin_pi_3};
  w[2] = {0, -1};
  w[3] = {-0.5, -sin_pi_3};
  w[4] = {-sin_pi_3, -0.5};

  if (raders_perm) {
    float2 temp[12] = {
        x[0],
        x[3],
        x[2],
        x[11],
        x[8],
        x[9],
        x[1],
        x[7],
        x[5],
        x[10],
        x[4],
        x[6]};
    radix6(temp, x);
    radix6(temp + 6, x + 6);
  } else {
    float2 temp[12] = {
        x[0],
        x[2],
        x[4],
        x[6],
        x[8],
        x[10],
        x[1],
        x[3],
        x[5],
        x[7],
        x[9],
        x[11]};
    radix6(temp, x);
    radix6(temp + 6, x + 6);
  }

  y[0] = x[0] + x[6];
  y[6] = x[0] - x[6];
  for (int t = 1; t < 6; t++) {
    float2 a = complex_mul(x[t + 6], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 6] = x[t] - a;
  }
}

METAL_FUNC void radix13(thread float2* x, thread float2* y) {
  // Raders Algorithm
  float2 inv = {1 / 12.0, -1 / 12.0};

  // fft
  radix12<true>(x + 1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(3.07497206, -1.88269669));
  y[3] = complex_mul_conj(y[3], float2(3.09912468, 1.84266823));
  y[4] = complex_mul_conj(y[4], float2(3.45084438, -1.04483161));
  y[5] = complex_mul_conj(y[5], float2(0.91083583, 3.48860690));
  y[6] = complex_mul_conj(y[6], float2(-3.60286363, 0.139189267));
  y[7] = complex_mul_conj(y[7], float2(3.60555128, 0));
  y[8] = complex_mul_conj(y[8], float2(3.60286363, 0.139189267));
  y[9] = complex_mul_conj(y[9], float2(0.91083583, -3.48860690));
  y[10] = complex_mul_conj(y[10], float2(-3.45084438, -1.04483161));
  y[11] = complex_mul_conj(y[11], float2(3.09912468, -1.84266823));
  y[12] = complex_mul_conj(y[12], float2(-3.07497206, -1.88269669));

  // ifft
  radix12<false>(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[7] = x[2] * inv + x[0];
  y[10] = x[3] * inv + x[0];
  y[5] = x[4] * inv + x[0];
  y[9] = x[5] * inv + x[0];
  y[11] = x[6] * inv + x[0];
  y[12] = x[7] * inv + x[0];
  y[6] = x[8] * inv + x[0];
  y[3] = x[9] * inv + x[0];
  y[8] = x[10] * inv + x[0];
  y[4] = x[11] * inv + x[0];
  y[2] = x[12] * inv + x[0];
}

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

#pragma clang loop unroll(full)
    for (int t = 2; t < radix; t++) {
      twiddle = complex_mul(twiddle, twiddle_1);
      x[t] = complex_mul(x[t], twiddle);
    }
  }

  radix_func(x, y);

#pragma clang loop unroll(full)
  for (int t = 0; t < radix; t++) {
    indices[t] = j + t * p;
  }
}

template <int radix, RadixFunc radix_func>
METAL_FUNC void radix_n_step(
    int i,
    thread int* p,
    int m,
    int n,
    int num_steps,
    thread float2* inputs,
    thread short* indices,
    thread float2* values,
    threadgroup float2* read_buf) {
  int m_r = n / radix;
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;

  int index = 0;
  int r_index = 0;
  for (int s = 0; s < num_steps; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          inputs[r] = read_buf[index + r * m_r];
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
          read_buf[indices[r_index]] = values[r_index];
        }
      }
    }

    // Wait until all threads have written back to threadgroup mem
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *p *= radix;
  }
}

#define RADIX_STEP(radix, radix_func, num_steps) \
  radix_n_step<radix, radix_func>(               \
      fft_idx, &p, m, n, num_steps, inputs, indices, values, read_buf);

#define RADER_STEP(radix, radix_func, num_steps) \
  radix_n_step<radix, radix_func>(               \
      fft_idx,                                   \
      &p,                                        \
      m,                                         \
      n - rader_m,                               \
      num_steps,                                 \
      inputs,                                    \
      indices,                                   \
      values,                                    \
      read_buf + rader_m);

#define RADIX_STEPS()                       \
  RADIX_STEP(2, radix2, radix_2_steps_);    \
  RADIX_STEP(3, radix3, radix_3_steps_);    \
  RADIX_STEP(4, radix4, radix_4_steps_);    \
  RADIX_STEP(5, radix5, radix_5_steps_);    \
  RADIX_STEP(7, radix7, radix_7_steps_);    \
  RADIX_STEP(8, radix8, radix_8_steps_);    \
  RADIX_STEP(11, radix11, radix_11_steps_); \
  RADIX_STEP(13, radix13, radix_13_steps_);

#define RADER_STEPS()                       \
  RADER_STEP(2, radix2, rader_2_steps_);    \
  RADER_STEP(3, radix3, rader_3_steps_);    \
  RADER_STEP(4, radix4, rader_4_steps_);    \
  RADER_STEP(5, radix5, rader_5_steps_);    \
  RADER_STEP(7, radix7, rader_7_steps_);    \
  RADER_STEP(8, radix8, rader_8_steps_);    \
  RADER_STEP(11, radix11, rader_11_steps_); \
  RADER_STEP(13, radix13, rader_13_steps_);

// Each FFT is computed entirely in shared GPU memory.
//
// N is decomposed into radix-n DFTs:
// e.g. 128 = 2 * 4 * 4 * 4
template <int tg_mem_size>
[[kernel]] void fft(
    const device float2* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  int batch_idx = elem.x * grid.y * n + tg_idx;

  // The number of the threads we're using for each DFT
  int m = grid.z;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* read_buf = &shared_in[tg_idx];

  // Account for possible extra threadgroups
  int grid_index = elem.x * grid.y + elem.y;
  if (grid_index >= batch_size) {
    return;
  }

  float2 inputs[MAX_RADIX];
  short indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

  for (int e = 0; e < elems_per_thread_; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    read_buf[index] = in[batch_idx + index];
    if (inv_) {
      read_buf[index].y = -read_buf[index].y;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // p is different
  // that must be breaking something

  int p = 1;
  RADIX_STEPS()

  // Write to device
  float2 inv_factor = {1.0f / n, -1.0f / n};
  for (int e = 0; e < elems_per_thread_; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    float2 elem = read_buf[index];
    if (inv_) {
      elem *= inv_factor;
    }
    out[batch_idx + index] = elem;
  }
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
  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  int batch_idx = elem.x * grid.y * n + tg_idx;

  // The number of the threads we're using for each DFT
  int m = grid.z;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* read_buf = &shared_in[tg_idx];
  // Account for possible extra threadgroups
  int grid_index = elem.x * grid.y + elem.y;
  if (grid_index >= batch_size) {
    return;
  }

  float2 inputs[MAX_RADIX];
  short indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

  // rader_m = n / rader_n;
  int rader_m = rader_m_;

  int index = 0;
  short g_q_index = 0;
  short g_q = 0;
  short out_index = 0;
  short diff_index = 0;
  short interleaved_index = 0;

  int max_index = n - rader_m - 1;

  for (int e = 0; e < elems_per_thread_; e++) {
    index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    g_q = raders_g_q[index / rader_m];
    read_buf[index + rader_m] =
        in[batch_idx + rader_m + (g_q - 1) * rader_m + index % rader_m];
    if (inv_) {
      read_buf[index + rader_m].y = -read_buf[index + rader_m].y;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  RADER_STEPS()

  // x_1 + ... + x_n is computed for us in the first FFT step so
  // we save it in the first rader_m indices of the array for later.
  int x_sum_index = metal::min(fft_idx, rader_m - 1);
  read_buf[x_sum_index] = read_buf[rader_m + x_sum_index * (rader_n - 1)];

  float2 temp[MAX_ELEMS_PER_THREAD];
  float2 inv = {1.0f, -1.0f};
  for (int e = 0; e < elems_per_thread_; e++) {
    index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    interleaved_index = index / rader_m + (index % rader_m) * (rader_n - 1);
    temp[e] = complex_mul(
        read_buf[rader_m + interleaved_index],
        raders_b_q[interleaved_index % (rader_n - 1)]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    read_buf[rader_m + index] = temp[e] * inv;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  p = 1;
  RADER_STEPS()

  short x_0_index =
      metal::min(fft_idx * elems_per_thread_ / (rader_n - 1), rader_m - 1);
  // We have to load two x_0s for each thread since sometimes
  // elems_per_thread_ crosses a boundary.
  // E.g. with n = 34, rader_n = 17, elems_per_thread_ = 4
  // 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 8 8
  // x x x x x x x x x x x x x x x x x - - - - - - - - - - - - - - - - -
  float2 x_0[2] = {in[batch_idx + x_0_index], in[batch_idx + x_0_index + 1]};
  if (inv_) {
    x_0[0].y = -x_0[0].y;
    x_0[1].y = -x_0[1].y;
  }

  float2 rader_inv_factor = {1.0f / (rader_n - 1), -1.0f / (rader_n - 1)};

  for (int e = 0; e < elems_per_thread_; e++) {
    index = metal::min(fft_idx * elems_per_thread_ + e, n - rader_m - 1);
    diff_index = index / (rader_n - 1) - x_0_index;
    temp[e] = read_buf[rader_m + index] * rader_inv_factor + x_0[diff_index];
  }

  // Use the sum of elements that was computed in the first FFT
  float2 x_sum = read_buf[x_0_index] + x_0[0];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    g_q_index = index % (rader_n - 1);
    g_q = raders_g_minus_q[g_q_index];
    out_index = index - g_q_index + g_q + (index / (rader_n - 1));
    read_buf[out_index] = temp[e];
  }

  read_buf[x_0_index * rader_n] = x_sum;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  p = rader_n;
  RADIX_STEPS()

  // Write to device
  float2 inv_factor = {1.0f / n, -1.0f / n};
  for (int e = 0; e < elems_per_thread_; e++) {
    index = metal::min(fft_idx * elems_per_thread_ + e, n - 1);
    float2 elem = read_buf[index];
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
  //   out = w_k * np.fft.ifft(np.fft.fft(w_k * in, n) * w_q)
  //
  // Where w_k and w_q are precomputed on CPU in high precision as:
  //   w_k = np.exp(-1j * np.pi / n * (np.arange(-n + 1, n) ** 2))
  //   w_q = np.fft.fft(1/w_k[-n:])

  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  int batch_idx = elem.x * grid.y * length + elem.y * length;

  int m = grid.z;

  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* read_buf = &shared_in[tg_idx];

  // Account for possible extra threadgroups
  int grid_index = elem.x * grid.y + elem.y;
  if (grid_index >= batch_size) {
    return;
  }

  float2 inputs[MAX_RADIX];
  short indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

  // load input into shared memory
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    if (index < length) {
      float2 elem = in[batch_idx + index];
      if (inv_) {
        elem.y = -elem.y;
      }
      read_buf[index] = complex_mul(elem, w_k[index]);
    } else {
      read_buf[index] = 0.0;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  RADIX_STEPS()

  float2 inv = float2(1.0f, -1.0f);
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    read_buf[index] = complex_mul(read_buf[index], w_q[index]) * inv;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ifft
  p = 1;
  RADIX_STEPS()

  float2 inv_factor = {1.0f / n, -1.0f / n};
  float2 inv_factor_overall = {1.0f / length, -1.0f / length};

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    if (index < length) {
      float2 elem = read_buf[index + length - 1] * inv_factor;
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
  // For rfft, we interleave batches of two real sequences into one complex one:
  // z_k = x_k + j.y_k
  // X_k = (Z_k + Z_(N-k)*) / 2
  // Y_k = -j * ((Z_k - Z_(N-k)*) / 2)

  int n_over_2 = (n / 2) + 1;

  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  int batch_idx = elem.x * grid.y * 2 * n + elem.y * 2 * n;
  int batch_idx_out = elem.x * grid.y * 2 * n_over_2 + elem.y * 2 * n_over_2;

  int m = grid.z;

  float2 inputs[MAX_RADIX];
  short indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

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
  threadgroup float2* read_buf = &shared_in[tg_idx];

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = metal::min(fft_idx + t * m, n - 1);
    read_buf[index].x = in[batch_idx + index];
    read_buf[index].y = in[batch_idx + index + next_in];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  RADIX_STEPS()

  float2 conj = {1, -1};
  float2 minus_j = {0, -1};
  for (int t = 0; t < elems_per_thread_ / 2; t++) {
    int index = metal::min(fft_idx + t * m, n_over_2 - 1);
    // Special case for first index of FFT
    // x_0 = z_0.real
    // y_0 = z_0.imag
    if (index == 0) {
      out[batch_idx_out + index] = {read_buf[index].x, 0};
      out[batch_idx_out + index + next_out] = {read_buf[index].y, 0};
    } else {
      float2 x_k = read_buf[index];
      float2 x_n_minus_k = read_buf[n - index] * conj;
      out[batch_idx_out + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx_out + index + next_out] =
          complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
  // Add in elements up to n/2 + 1
  int num_left = n_over_2 - (elems_per_thread_ / 2 * m);
  if (fft_idx < num_left) {
    int index = metal::min(fft_idx + elems_per_thread_ / 2 * m, n_over_2 - 1);
    float2 x_k = read_buf[index];
    float2 x_n_minus_k = read_buf[n - index] * conj;
    out[batch_idx_out + index] = (x_k + x_n_minus_k) / 2;
    out[batch_idx_out + index + next_out] =
        complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
  }
}

#define instantiate_fft(tg_mem_size)                              \
  template [[host_name("fft_mem_" #tg_mem_size)]] [[kernel]] void \
  fft<tg_mem_size>(                                               \
      const device float2* in [[buffer(0)]],                      \
      device float2* out [[buffer(1)]],                           \
      constant const int& n,                                      \
      constant const int& batch_size,                             \
      uint3 elem [[thread_position_in_grid]],                     \
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

#define instantiate_rfft(tg_mem_size)                              \
  template [[host_name("rfft_mem_" #tg_mem_size)]] [[kernel]] void \
  rfft<tg_mem_size>(                                               \
      const device float* in [[buffer(0)]],                        \
      device float2* out [[buffer(1)]],                            \
      constant const int& n,                                       \
      constant const int& batch_size,                              \
      uint3 elem [[thread_position_in_grid]],                      \
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

// clang-format off
#define instantiate_ffts(tg_mem_size)                        \
  instantiate_fft(tg_mem_size) \
  instantiate_rfft(tg_mem_size) \
  instantiate_rader(tg_mem_size) \
  instantiate_bluestein(tg_mem_size)

// It's substantially faster to statically define the
// threadgroup memory size rather than using
// `setThreadgroupMemoryLength` on the compute encoder.
// For non-power of 2 sizes we round up the shared memory.
instantiate_ffts(4)
instantiate_ffts(8)
instantiate_ffts(16)
instantiate_ffts(32)
instantiate_ffts(64)
instantiate_ffts(128)
instantiate_ffts(256)
instantiate_ffts(512)
instantiate_ffts(1024)
instantiate_ffts(2048)
// 4096 is the max that will fit into 32KB of threadgroup memory.
instantiate_ffts(4096) // clang-format on
