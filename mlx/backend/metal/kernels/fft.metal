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

// In the worst case we hae 13 elems per thread
#define MAX_OUTPUT_SIZE 22

// Specialize for a particular value of N at runtime
constant bool inv_ [[function_constant(0)]];
constant bool is_power_of_2_ [[function_constant(1)]];
constant int elems_per_thread_ [[function_constant(2)]];
// Signal which radices we need to read/write to global memory on
constant int first_radix_ [[function_constant(3)]];
constant int last_radix_ [[function_constant(4)]];
constant int first_rader_ [[function_constant(5)]];
constant int last_rader_ [[function_constant(6)]];
// Stockham steps
constant int radix_13_steps_ [[function_constant(7)]];
constant int radix_11_steps_ [[function_constant(8)]];
constant int radix_7_steps_ [[function_constant(9)]];
constant int radix_5_steps_ [[function_constant(10)]];
constant int radix_4_steps_ [[function_constant(11)]];
constant int radix_3_steps_ [[function_constant(12)]];
constant int radix_2_steps_ [[function_constant(13)]];
// Rader steps
constant int rader_13_steps_ [[function_constant(14)]];
constant int rader_11_steps_ [[function_constant(15)]];
constant int rader_7_steps_ [[function_constant(16)]];
constant int rader_5_steps_ [[function_constant(17)]];
constant int rader_4_steps_ [[function_constant(18)]];
constant int rader_3_steps_ [[function_constant(19)]];
constant int rader_2_steps_ [[function_constant(20)]];

float2 complex_mul(float2 a, float2 b) {
  float2 c = {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
  return c;
}

float2 get_twiddle(int k, int p) {
  float theta = -2.0f * k * M_PI_F / p;

  float2 twiddle = {metal::fast::cos(theta), metal::fast::sin(theta)};
  return twiddle;
}

typedef void (*RadixFunc)(thread float2*, thread float2*);

template <int radix, RadixFunc radix_func>
void radix_n(
    int i,
    int p,
    thread float2* x,
    thread int* indices,
    thread float2* y) {
  // i: the index in the overall DFT that we're processing.
  // p: the size of the DFTs we're merging at this step.
  // m: how many threads are working on this DFT.

  int k = i % p;
  // int k = i & (p - 1);
  int j = (i / p) * radix * p + k;
  // int j = ((i - k) << 2) + k;

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

// Radix kernels
//
// We provide optimized, single threaded Radix codelets
// for n=2,3,4,5,6,7,10,11,12,13.
//
// For n=2,3,4,5,6 we hand write the codelets.
// For n=10,12 we combine radix2 and radix5/6 codelets.
// For n=7,11,13 we use Rader's algorithm which decomposes
// them into (n-1)=6,10,12 codelets.

void radix2(thread float2* x, thread float2* y) {
  y[0] = x[0] + x[1];
  y[1] = x[0] - x[1];
}

void radix3(thread float2* x, thread float2* y) {
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

void radix4(thread float2* x, thread float2* y) {
  float2 minus_i = {0, -1};

  float2 z_0 = x[0] + x[2];
  float2 z_1 = x[0] - x[2];
  float2 z_2 = x[1] + x[3];
  float2 z_3 = complex_mul(x[1] - x[3], minus_i);

  y[0] = z_0 + z_2;
  y[1] = z_1 + z_3;
  y[2] = z_0 - z_2;
  y[3] = z_1 - z_3;
}

void radix5(thread float2* x, thread float2* y) {
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

void radix6(thread float2* x, thread float2* y) {
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

void radix7(thread float2* x, thread float2* y) {
  // Rader's algorithm
  float2 b_q[6];
  b_q[0] = {-1, 0};
  b_q[1] = {2.44013336, -1.02261879};
  b_q[2] = {2.37046941, -1.17510629};
  b_q[3] = {0, -2.64575131};
  b_q[4] = {b_q[2].x, -b_q[2].y};
  b_q[5] = {-b_q[1].x, b_q[1].y};

  float2 in1[6] = {x[1], x[3], x[2], x[6], x[4], x[5]};
  radix6(in1, y + 1);
  float2 x_sum = y[1];

  float2 conj = {1, -1};
  float2 inv = {1 / 6.0, -1 / 6.0};

#pragma clang loop unroll(full)
  for (int t = 0; t < 6; t++) {
    y[t + 1] = complex_mul(y[t + 1], b_q[t]) * conj;
  }

  radix6(y + 1, x + 1);

  // Inv Rader permutation
  int perm[6] = {1, 5, 4, 6, 2, 3};
  y[0] = x_sum + x[0];
#pragma clang loop unroll(full)
  for (int t = 0; t < 6; t++) {
    y[perm[t]] = x[t + 1] * inv + x[0];
  }
}

// void radix8(thread float2* x, thread float2* y) {
//   float cos_pi_4 = 0.7071067811865476;
// }

void radix10(thread float2* x, thread float2* y) {
  float2 w[4];
  w[0] = {0.8090169943749475, -0.5877852522924731};
  w[1] = {0.30901699437494745, -0.9510565162951535};
  w[2] = {-w[1].x, w[1].y};
  w[3] = {-w[0].x, w[0].y};

  float2 temp[10] = {
      x[0], x[2], x[4], x[6], x[8], x[1], x[3], x[5], x[7], x[9]};
  radix5(temp, x);
  radix5(temp + 5, x + 5);

  y[0] = x[0] + x[5];
  y[5] = x[0] - x[5];
  for (int t = 1; t < 5; t++) {
    float2 a = complex_mul(x[t + 5], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 5] = x[t] - a;
  }
}

void radix11(thread float2* x, thread float2* y) {
  // Raders Algorithm
  float2 b_q[10];
  b_q[0] = {-1, 0};
  b_q[1] = {0.955301878, -3.17606649};
  b_q[2] = {2.63610556, 2.01269656};
  b_q[3] = {2.54127802, 2.13117479};
  b_q[4] = {2.07016210, 2.59122150};
  b_q[5] = {0, -3.31662479};
  b_q[6] = {b_q[4].x, -b_q[4].y};
  b_q[7] = {-b_q[3].x, b_q[3].y};
  b_q[8] = {b_q[2].x, -b_q[2].y};
  b_q[9] = {-b_q[1].x, b_q[1].y};

  float2 in1[10] = {
      x[1], x[2], x[4], x[8], x[5], x[10], x[9], x[7], x[3], x[6]};
  radix10(in1, y + 1);
  float2 x_sum = y[1];

  float2 conj = {1, -1};
  float2 inv = {1 / 10.0, -1 / 10.0};

#pragma clang loop unroll(full)
  for (int t = 0; t < 10; t++) {
    y[t + 1] = complex_mul(y[t + 1], b_q[t]) * conj;
  }

  radix10(y + 1, x + 1);

  int perm[10] = {1, 6, 3, 7, 9, 10, 5, 8, 4, 2};
  y[0] = x_sum + x[0];
#pragma clang loop unroll(full)
  for (int t = 0; t < 10; t++) {
    y[perm[t]] = x[t + 1] * inv + x[0];
  }
}

void radix12(thread float2* x, thread float2* y) {
  float2 w[6];
  float sin_pi_3 = 0.8660254037844387;
  w[0] = {sin_pi_3, -0.5};
  w[1] = {0.5, -sin_pi_3};
  w[2] = {0, -1};
  w[3] = {-0.5, -sin_pi_3};
  w[4] = {-sin_pi_3, -0.5};

  float2 temp[12] = {
      x[0], x[2], x[4], x[6], x[8], x[10], x[1], x[3], x[5], x[7], x[9], x[11]};
  radix6(temp, x);
  radix6(temp + 6, x + 6);

  y[0] = x[0] + x[6];
  y[6] = x[0] - x[6];
  for (int t = 1; t < 6; t++) {
    float2 a = complex_mul(x[t + 6], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 6] = x[t] - a;
  }
}

void radix13(thread float2* x, thread float2* y) {
  // Raders Algorithm
  float2 b_q[12];
  b_q[0] = {-1, 0};
  b_q[1] = {3.07497206, -1.88269669};
  b_q[2] = {3.09912468, 1.84266823};
  b_q[3] = {3.45084438, -1.04483161};
  b_q[4] = {0.91083583, 3.48860690};
  b_q[5] = {-3.60286363, 0.139189267};
  b_q[6] = {3.60555128, 0};
  b_q[7] = {-b_q[5].x, b_q[5].y};
  b_q[8] = {b_q[4].x, -b_q[4].y};
  b_q[9] = {-b_q[3].x, b_q[3].y};
  b_q[10] = {b_q[2].x, -b_q[2].y};
  b_q[11] = {-b_q[1].x, b_q[1].y};

  float2 in1[12] = {
      x[1],
      x[2],
      x[4],
      x[8],
      x[3],
      x[6],
      x[12],
      x[11],
      x[9],
      x[5],
      x[10],
      x[7]};
  radix12(in1, y + 1);

  float2 x_sum = y[1];

  float2 conj = {1, -1};
  float2 inv = {1 / 12.0, -1 / 12.0};

#pragma clang loop unroll(full)
  for (int t = 0; t < 12; t++) {
    y[t + 1] = complex_mul(y[t + 1], b_q[t]) * conj;
  }

  radix12(y + 1, x + 1);

  int perm[12] = {1, 7, 10, 5, 9, 11, 12, 6, 3, 8, 4, 2};
  y[0] = x_sum + x[0];
#pragma clang loop unroll(full)
  for (int t = 0; t < 12; t++) {
    y[perm[t]] = x[t + 1] * inv + x[0];
  }
}

template <int radix, RadixFunc radix_func>
void radix_n_step(
    int i,
    thread int* p,
    int m,
    int n,
    int batch_idx,
    int num_steps,
    threadgroup float2* read_buf,
    const device float2* in,
    device float2* out) {
  // Thread local memory for inputs and outputs to the radix codelet
  float2 inputs[radix];
  int indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

  int m_r = n / radix;
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;

  for (int s = 0; s < num_steps; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          if (s == 0 && first_rader_ == 0 && radix == first_radix_) {
            inputs[r] = in[batch_idx + index + r * m_r];
          } else {
            inputs[r] = read_buf[index + r * m_r];
          }
        }
        radix_n<radix, radix_func>(
            index, *p, inputs, indices + t * radix, values + t * radix);
      }
    }

    // Wait until all threads have read their inputs into thread local mem
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int r_index = t * radix + r;
          if (s == num_steps - 1 && radix == last_radix_) {
            out[batch_idx + indices[r_index]] = values[r_index];
          } else {
            read_buf[indices[r_index]] = values[r_index];
          }
        }
      }
    }

    // Wait until all threads have written back to threadgroup mem
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *p *= radix;
  }
}

template <int radix, RadixFunc radix_func>
void rader_n_step_forward(
    int i,
    thread int* p,
    int m,
    int rader_m,
    int n,
    int batch_idx,
    int num_steps,
    threadgroup float2* read_buf,
    const device float2* in,
    device float2* out,
    const device short* raders_g_q,
    const device float2* raders_b_q) {
  // Thread local memory for inputs and outputs to the radix codelet
  float2 inputs[radix];
  int indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

  float2 inv = {1.0f, -1.0f};
  int m_r = (n - rader_m) / radix;
  int rader_n = n / rader_m;
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < num_steps; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int m_index = index + r * m_r;
          if (s == 0 && first_rader_ == radix) {
            /* Rader permutation for the input */
            short g_q = raders_g_q[m_index / rader_m];
            inputs[r] =
                in[batch_idx + rader_m + (g_q - 1) * rader_m + index % rader_m];
          } else {
            inputs[r] = read_buf[rader_m + m_index];
          }
        }
        radix_n<radix, radix_func>(
            index, *p, inputs, indices + t * radix, values + t * radix);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int r_index = r + t * radix;
          if (s == num_steps - 1 && last_rader_ == radix) {
            read_buf[rader_m + indices[r_index]] =
                complex_mul(
                    values[r_index],
                    raders_b_q[indices[r_index] % (rader_n - 1)]) *
                inv;
            // Fill in x_0
            if (indices[r_index] % (rader_n - 1) == 0) {
              read_buf[indices[r_index] / (rader_n - 1)] = values[r_index];
            }
          } else {
            read_buf[rader_m + indices[r_index]] = values[r_index];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *p *= radix;
  }
}

template <int radix, RadixFunc radix_func>
void rader_n_step_backward(
    int i,
    thread int* p,
    int m,
    int rader_m,
    int n,
    int batch_idx,
    int num_steps,
    threadgroup float2* read_buf,
    const device float2* in,
    device float2* out,
    const device short* raders_g_minus_q) {
  // Thread local memory for inputs and outputs to the radix codelet
  float2 inputs[radix];
  int indices[MAX_OUTPUT_SIZE];
  float2 values[MAX_OUTPUT_SIZE];

  int m_r = (n - rader_m) / radix;
  int rader_n = n / rader_m;
  float2 inv_factor = {1.0f / (rader_n - 1), -1.0f / (rader_n - 1)};
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < num_steps; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int m_index = index + r * m_r;
          if (s == 0 && first_rader_ == radix) {
            // We get the data uninterleaved from the output of the forward FFT
            inputs[r] = read_buf
                [rader_m + m_index / rader_m +
                 (m_index % rader_m) * (rader_n - 1)];
          } else {
            inputs[r] = read_buf[rader_m + m_index];
          }
        }
        radix_n<radix, radix_func>(
            index, *p, inputs, indices + t * radix, values + t * radix);
      }
    }

    bool last_step = s == num_steps - 1 && last_rader_ == radix;
    float2 x_sum[13];

    if (last_step) {
      for (int t = 0; t < max_radices_per_thread; t++) {
        int index = i + t * m;
        for (int e = 0; e < elems_per_thread_; e++) {
          int e_index = index * elems_per_thread_ + e;
          if (e_index < rader_m) {
            x_sum[e] = read_buf[e_index] + in[batch_idx + e_index];
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int r_index = r + t * radix;
          if (last_step) {
            // Rader permutation for the output
            int g_q_index = indices[r_index] % (rader_n - 1);
            short g_q = raders_g_minus_q[g_q_index];
            // we've been doing the rader's n - 1 fft/ifft like this:
            // _ _ x x x x x x x x
            // we need to rearrange to this so we can go straight into normal
            // radix steps: _ x x x x _ x x x x
            int out_index = indices[r_index] - g_q_index + g_q +
                (indices[r_index] / (rader_n - 1));
            float2 x_0 = in[batch_idx + out_index / rader_n];
            if (last_radix_ == 0) {
              out[batch_idx + out_index] = values[r_index] * inv_factor + x_0;
            } else {
              read_buf[out_index] = values[r_index] * inv_factor + x_0;
            }
          } else {
            read_buf[rader_m + indices[r_index]] = values[r_index];
          }
        }
      }
    }

    if (last_step) {
      for (int t = 0; t < max_radices_per_thread; t++) {
        int index = i + t * m;
        for (int e = 0; e < elems_per_thread_; e++) {
          int e_index = index * elems_per_thread_ + e;
          if (e_index < rader_m) {
            if (last_radix_ == 0) {
              out[batch_idx + e_index * rader_n] = x_sum[e];
            } else {
              read_buf[e_index * rader_n] = x_sum[e];
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    *p *= radix;
  }
}

#define RADIX_STEP(radix, radix_func, num_steps) \
  radix_n_step<radix, radix_func>(               \
      i, &p, m, n, batch_idx, num_steps, read_buf, in, out);

#define RADER_FORWARD_STEP(radix, radix_func, num_steps) \
  rader_n_step_forward<radix, radix_func>(               \
      i,                                                 \
      &p,                                                \
      m,                                                 \
      rader_m,                                           \
      n,                                                 \
      batch_idx,                                         \
      num_steps,                                         \
      read_buf,                                          \
      in,                                                \
      out,                                               \
      raders_g_q,                                        \
      raders_b_q);

#define RADER_BACKWARD_STEP(radix, radix_func, num_steps) \
  rader_n_step_backward<radix, radix_func>(               \
      i,                                                  \
      &p,                                                 \
      m,                                                  \
      rader_m,                                            \
      n,                                                  \
      batch_idx,                                          \
      num_steps,                                          \
      read_buf,                                           \
      in,                                                 \
      out,                                                \
      raders_g_minus_q);

// Each FFT is computed entirely in shared GPU memory.
//
// N is decomposed into radix-n DFTs:
// e.g. 128 = 2 * 4 * 4 * 4
template <int tg_mem_size>
[[kernel]] void fft(
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
  int i = elem.z;
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

  int rader_m = n / rader_n;

  int p = 1;
  RADER_FORWARD_STEP(2, radix2, rader_2_steps_);
  RADER_FORWARD_STEP(3, radix3, rader_3_steps_);
  RADER_FORWARD_STEP(4, radix4, rader_4_steps_);
  RADER_FORWARD_STEP(5, radix5, rader_5_steps_);
  RADER_FORWARD_STEP(7, radix7, rader_7_steps_);
  RADER_FORWARD_STEP(11, radix11, rader_11_steps_);
  RADER_FORWARD_STEP(13, radix13, rader_13_steps_);

  p = 1;
  RADER_BACKWARD_STEP(2, radix2, rader_2_steps_);
  RADER_BACKWARD_STEP(3, radix3, rader_3_steps_);
  RADER_BACKWARD_STEP(4, radix4, rader_4_steps_);
  RADER_BACKWARD_STEP(5, radix5, rader_5_steps_);
  RADER_BACKWARD_STEP(7, radix7, rader_7_steps_);
  RADER_BACKWARD_STEP(11, radix11, rader_11_steps_);
  RADER_BACKWARD_STEP(13, radix13, rader_13_steps_);

  p = rader_n;
  RADIX_STEP(2, radix2, radix_2_steps_);
  RADIX_STEP(3, radix3, radix_3_steps_);
  RADIX_STEP(4, radix4, radix_4_steps_);
  RADIX_STEP(5, radix5, radix_5_steps_);
  RADIX_STEP(7, radix7, radix_7_steps_);
  RADIX_STEP(11, radix11, radix_11_steps_);
  RADIX_STEP(13, radix13, radix_13_steps_);
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
    int index = fft_idx + t * m;
    read_buf[index].x = in[batch_idx + index];
    read_buf[index].y = in[batch_idx + index + next_in];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // perform_fft(fft_idx, n, m, &read_buf, &write_buf);

  float2 conj = {1, -1};
  float2 minus_j = {0, -1};
  for (int t = 0; t < elems_per_thread_ / 2; t++) {
    int index = fft_idx + t * m;
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
    int index = fft_idx + elems_per_thread_ / 2 * m;
    float2 x_k = read_buf[index];
    float2 x_n_minus_k = read_buf[n - index] * conj;
    out[batch_idx_out + index] = (x_k + x_n_minus_k) / 2;
    out[batch_idx_out + index + next_out] =
        complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
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

  // perform_fft(fft_idx, n, m, &read_buf, &write_buf);

  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    read_buf[index] = complex_mul(read_buf[index], w_q[index]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ifft
  for (int t = 0; t < elems_per_thread_; t++) {
    read_buf[fft_idx + t * m].y = -read_buf[fft_idx + t * m].y;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // perform_fft(fft_idx, n, m, &read_buf, &write_buf);

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

#define instantiate_fft(tg_mem_size)                              \
  template [[host_name("fft_mem_" #tg_mem_size)]] [[kernel]] void \
  fft<tg_mem_size>(                                               \
      const device float2* in [[buffer(0)]],                      \
      device float2* out [[buffer(1)]],                           \
      const device float2* raders_b_q [[buffer(2)]],              \
      const device short* raders_g_q [[buffer(3)]],               \
      const device short* raders_g_minus_q [[buffer(4)]],         \
      constant const int& n,                                      \
      constant const int& batch_size,                             \
      constant const int& rader_n,                                \
      uint3 elem [[thread_position_in_grid]],                     \
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

#define instantiate_ffts(tg_mem_size)                        \
  instantiate_fft(tg_mem_size) instantiate_rfft(tg_mem_size) \
      instantiate_bluestein(tg_mem_size)

// It's substantially faster to statically define the
// threadgroup memory size rather than using
// `setThreadgroupMemoryLength` on the compute encoder.
// For non-power of 2 sizes we round up the shared memory.
instantiate_ffts(4) instantiate_ffts(8) instantiate_ffts(16)
    instantiate_ffts(32) instantiate_ffts(64) instantiate_ffts(128)
        instantiate_ffts(256) instantiate_ffts(512) instantiate_ffts(1024)
            instantiate_ffts(2048)
    // 4096 is the max that will fit into 32KB of threadgroup memory.
    instantiate_ffts(4096)
