// Copyright © 2024 Apple Inc.

// Metal FFT using Stockham's algorithm
//
// References:
// - VkFFT (https://github.com/DTolm/VkFFT)
// - Eric Bainville's excellent page (http://www.bealto.com/gpu-fft.html)

#include <metal_math>
#include <metal_common>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;


// Specialize for a particular value of N at runtime
constant bool inv_ [[function_constant(0)]];
constant bool is_power_of_2_ [[function_constant(1)]];
constant int elems_per_thread_ [[function_constant(2)]];
constant int radix_7_steps_ [[function_constant(3)]];
constant int radix_5_steps_ [[function_constant(4)]];
constant int radix_4_steps_ [[function_constant(5)]];
constant int radix_3_steps_ [[function_constant(6)]];
constant int radix_2_steps_ [[function_constant(7)]];
constant int rader_7_steps_ [[function_constant(8)]];
constant int rader_5_steps_ [[function_constant(9)]];
constant int rader_4_steps_ [[function_constant(10)]];
constant int rader_3_steps_ [[function_constant(11)]];
constant int rader_2_steps_ [[function_constant(12)]];

float2 complex_mul(float2 a, float2 b) {
  float2 c = {
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  };
  return c;
}

typedef void(*RadixFunc)(int, int, thread float2*, thread int*, thread float2*);

template <int radix, RadixFunc radix_func>
void radix_n_step(
    int i, int p, int m, int n, int batch_idx,
    threadgroup float2* read_buf,
    const device float2* in, device float2* out) {

  // Thread local memory for inputs and outputs to the radix codelet
  float2 inputs[radix];
  int indices[radix];
  float2 values[radix];

  int m_r = n / radix;
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;

  for (int s = 0; s < 1; s++) {
    for (int t = 0; t < 1; t++) {
      int index = i + t * m;
      for (int r = 0; r < radix; r++) {
        if (s == 0 && false) {
          inputs[r] = in[batch_idx + index + r*m_r];
        } else {
          inputs[r] = read_buf[index + r*m_r];
        }
      }
      radix_func(index, p, inputs, indices + t*radix, values + t*radix);
    }

    // Wait until all threads have read their inputs into thread local mem
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < 1; t++) {
      for (int r = 0; r < radix; r++) {
        int r_index = t*radix + r;
        if (s == 0 && true) {
          out[batch_idx + indices[r_index]] = values[r_index];
        } else {
          read_buf[indices[r_index]] = values[r_index];
        }
      }
    }

    // Wait until all threads have written back to threadgroup mem
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p *= radix;
  }
}

template <int radix, RadixFunc radix_func>
void rader_n_step_forward(
    int i, int p, int m, int rader_m, int n, int batch_idx,
    float2 x_0,
    threadgroup float2* read_buf,
    const device float2* in, device float2* out,
    const device short* raders_g_q, const device float2* raders_b_q) {
  int first_rader = 2;
  int last_rader = 2;

  // Thread local memory for inputs and outputs to the radix codelet
  float2 inputs[radix];
  int indices[radix];
  float2 values[radix];

  float2 inv = {1.0f, -1.0f};
  int m_r = (n - rader_m) / radix;
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < 2; s++) {
    for (int t = 0; t < 1; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int m_index = index + r*m_r;
          if (s == 0 && first_rader == radix) {
            /* Rader permutation for the input */
            short g_q = raders_g_q[m_index / rader_m];
            inputs[r] = in[batch_idx + rader_m + (g_q - 1) * rader_m + index % rader_m];
          } else {
            inputs[r] = read_buf[rader_m + m_index];
          }
        }
        radix_func(index, p, inputs, indices + t*radix, values + t*radix);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int r_index = r + t*radix;
          if (s == 2 - 1 && last_rader == radix) {
            read_buf[rader_m + indices[r_index]] = complex_mul(values[r_index], raders_b_q[indices[r_index] % m_r]) * inv;
            // Fill in x_0
            if (index % rader_m == 0 && r == 0) {
              read_buf[index / rader_m] = values[r_index] + x_0;
            }
          } else {
            read_buf[rader_m + indices[r_index]] = values[r_index];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p *= radix;
  }
}

template <int radix, RadixFunc radix_func>
void rader_n_step_backward(
    int i, int p, int m, int rader_m, int n, int batch_idx,
    float2 x_0,
    threadgroup float2* read_buf,
    device float2* out,
    const device short* raders_g_minus_q) {
  int first_rader = 2;
  int last_rader = 2;

  // Thread local memory for inputs and outputs to the radix codelet
  float2 inputs[radix];
  int indices[radix];
  float2 values[radix];

  int m_r = (n - rader_m) / radix;
  float2 inv_factor = {1.0f / m_r, -1.0f / m_r};
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;
  for (int s = 0; s < 2; s++) {
    for (int t = 0; t < 1; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int m_index = index + r*m_r;
          if (s == 0 && first_rader == radix) {
            // We get the data uninterleaved from the output of the forward FFT
            inputs[r] = read_buf[rader_m + (i % rader_m)*m_r + i / rader_m + rader_m*r];
          } else {
            inputs[r] = read_buf[rader_m + m_index];
          }
        }
        radix_func(index, p, inputs, indices + t*radix, values + t*radix);
      } else if (index < m && s == 0) {
        for (int r = 0; r < radix; r++) {
          inputs[r] = read_buf[r];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < max_radices_per_thread; t++) {
      int index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          int r_index = r + t*radix;
          if (s == 2 - 1 && last_rader == radix) {
            /* Rader permutation for the output */
            int g_index = raders_g_minus_q[indices[r_index] % m_r] + (index / rader_m) * (n / rader_m);
            read_buf[g_index] = values[r_index] * inv_factor + x_0;
          } else {
            read_buf[rader_m + indices[r_index]] = values[r_index];
          }
        }
      } else if (index < m && s == 2 - 1) {
        for (int r = 0; r < radix; r++) {
          read_buf[r * (n / rader_m)] = inputs[r];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p *= radix;
  }
} 

// void radix_fft(
//     int i, int p, int m, int n, int batch_idx,
//     threadgroup float2* read_buf,
//     const device float2* in, device float2* out) {
//   radix_n_step<2, radix2>(i, p, m, n, batch_idx, read_buf, in, out);
//   radix_n_step<3, radix3>(i, p, m, n, batch_idx, read_buf, in, out);
//   radix_n_step<4, radix4>(i, p, m, n, batch_idx, read_buf, in, out);
//   radix_n_step<5, radix5>(i, p, m, n, batch_idx, read_buf, in, out);
//   radix_n_step<7, radix7>(i, p, m, n, batch_idx, read_buf, in, out);
// }

float2 get_twiddle(int k, int p) {
  float theta = -2.0f * k * M_PI_F / p;

  float2 twiddle = {
      metal::fast::cos(theta),
      metal::fast::sin(theta)
  };
  return twiddle;
}

void radix2(int i, int p, thread float2* inputs, thread int* indices, thread float2* values) {
  // i: the index in the overall DFT that we're processing.
  // p: the size of the DFTs we're merging at this step.
  // m: how many threads are working on this DFT.

  float2 x_0 = inputs[0];
  float2 x_1 = inputs[1];

  // The index within this sub-DFT
  // int k = i & (p - 1);
  int k = i % p;

  float2 twiddle = get_twiddle(k, 2*p);

  float2 z = complex_mul(x_1, twiddle);

  float2 y_0 = x_0 + z;
  float2 y_1 = x_0 - z;

  // int j = (i << 1) - k;
  int j = (i / p) * 2 * p + k;

  indices[0] = j;
  indices[1] = j + p;
  values[0] = y_0;
  values[1] = y_1;
}

void radix3(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  float2 w_1 = {-0.5, -0.8660254037844387};
  float2 w_2 = {-0.5, 0.8660254037844384};

  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];
  float2 x_2 = read_buf[i + 2*m];

  int k = i % p;
  int j = (i / p) * 3 * p + k;

  float2 twiddle_1 = get_twiddle(k, 3*p);
  float2 twiddle_2 = complex_mul(twiddle_1, twiddle_1);

  x_1 = complex_mul(x_1, twiddle_1);
  x_2 = complex_mul(x_2, twiddle_2);

  float2 y_0 = x_0 + x_1 + x_2;
  float2 y_1 = x_0 + complex_mul(x_1, w_1) + complex_mul(x_2, w_2);
  float2 y_2 = x_0 + complex_mul(x_1, w_2) + complex_mul(x_2, w_1);

  write_buf[j] = y_0;
  write_buf[j + p] = y_1;
  write_buf[j + 2*p] = y_2;
}

void radix4(int i, int p, int m, thread float2* inputs, thread int* indices, thread float2* values) {
  float2 x_0 = inputs[0];
  float2 x_1 = inputs[1];
  float2 x_2 = inputs[2];
  float2 x_3 = inputs[3];

  // We use faster bit shifting ops when n is a power of 2
  int k;
  if (is_power_of_2_) {
    k = i & (p - 1);
  } else {
    k = i % p;
  }

  if (p > 1) {
    float2 twiddle = get_twiddle(k, 4*p);
    // e^a * e^b = e^(a + b)
    float2 twiddle_2 = complex_mul(twiddle, twiddle);
    float2 twiddle_3 = complex_mul(twiddle, twiddle_2);

    x_1 = complex_mul(x_1, twiddle);
    x_2 = complex_mul(x_2, twiddle_2);
    x_3 = complex_mul(x_3, twiddle_3);
  }

  float2 minus_i = {0, -1};

  // Hard coded twiddle factors for DFT4
  float2 z_0 = x_0 + x_2;
  float2 z_1 = x_0 - x_2;
  float2 z_2 = x_1 + x_3;
  float2 z_3 = complex_mul(x_1 - x_3, minus_i);

  float2 y_0 = z_0 + z_2;
  float2 y_1 = z_1 + z_3;
  float2 y_2 = z_0 - z_2;
  float2 y_3 = z_1 - z_3;

  int j;
  if (is_power_of_2_) {
    j = ((i - k) << 2) + k;
  } else {
    j = (i / p) * 4 * p + k;
  }

  indices[0] = j;
  indices[1] = j + p;
  indices[2] = j + 2*p;
  indices[3] = j + 3*p;
  values[0] = y_0;
  values[1] = y_1;
  values[2] = y_2;
  values[3] = y_3;
}

void radix5(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  float2 w_1 = {0.30901699437494745, -0.9510565162951535};
  float2 w_2 = {-0.8090169943749473, -0.5877852522924732};
  float2 w_3 = {-0.8090169943749475, 0.587785252292473};
  float2 w_4 = {0.30901699437494723, 0.9510565162951536};

  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];
  float2 x_2 = read_buf[i + 2*m];
  float2 x_3 = read_buf[i + 3*m];
  float2 x_4 = read_buf[i + 4*m];

  int k = i % p;
  int j = (i / p) * 5 * p + k;

  float2 twiddle_1 = get_twiddle(k, 5*p);
  float2 twiddle_2 = complex_mul(twiddle_1, twiddle_1);
  float2 twiddle_3 = complex_mul(twiddle_1, twiddle_2);
  float2 twiddle_4 = complex_mul(twiddle_1, twiddle_3);

  x_1 = complex_mul(x_1, twiddle_1);
  x_2 = complex_mul(x_2, twiddle_2);
  x_3 = complex_mul(x_3, twiddle_3);
  x_4 = complex_mul(x_4, twiddle_4);

  float2 y_0 = x_0 + x_1 + x_2 + x_3 + x_4;
  float2 y_1 = x_0 + complex_mul(x_1, w_1) + complex_mul(x_2, w_2) + complex_mul(x_3, w_3) + complex_mul(x_4, w_4);
  float2 y_2 = x_0 + complex_mul(x_1, w_2) + complex_mul(x_2, w_4) + complex_mul(x_3, w_1) + complex_mul(x_4, w_3);
  float2 y_3 = x_0 + complex_mul(x_1, w_3) + complex_mul(x_2, w_1) + complex_mul(x_3, w_4) + complex_mul(x_4, w_2);
  float2 y_4 = x_0 + complex_mul(x_1, w_4) + complex_mul(x_2, w_3) + complex_mul(x_3, w_2) + complex_mul(x_4, w_1);

  write_buf[j] = y_0;
  write_buf[j + p] = y_1;
  write_buf[j + 2*p] = y_2;
  write_buf[j + 3*p] = y_3;
  write_buf[j + 4*p] = y_4;
}

void radix7(int i, int p, int m, threadgroup float2* read_buf, threadgroup float2* write_buf) {
  float2 w_1 = {0.6234898018587336, -0.7818314824680298};
  float2 w_2 = {-0.22252093395631434, -0.9749279121818236};
  float2 w_3 = {-0.900968867902419, -0.43388373911755823};
  float2 w_4 = {-0.9009688679024191, 0.433883739117558};
  float2 w_5 = {-0.2225209339563146, 0.9749279121818236};
  float2 w_6 = {0.6234898018587334, 0.7818314824680299};

  float2 x_0 = read_buf[i];
  float2 x_1 = read_buf[i + m];
  float2 x_2 = read_buf[i + 2*m];
  float2 x_3 = read_buf[i + 3*m];
  float2 x_4 = read_buf[i + 4*m];
  float2 x_5 = read_buf[i + 5*m];
  float2 x_6 = read_buf[i + 6*m];

  int k = i % p;
  int j = (i / p) * 7 * p + k;

  float2 twiddle_1 = get_twiddle(k, 7*p);
  float2 twiddle_2 = complex_mul(twiddle_1, twiddle_1);
  float2 twiddle_3 = complex_mul(twiddle_1, twiddle_2);
  float2 twiddle_4 = complex_mul(twiddle_1, twiddle_3);
  float2 twiddle_5 = complex_mul(twiddle_1, twiddle_4);
  float2 twiddle_6 = complex_mul(twiddle_1, twiddle_5);

  x_1 = complex_mul(x_1, twiddle_1);
  x_2 = complex_mul(x_2, twiddle_2);
  x_3 = complex_mul(x_3, twiddle_3);
  x_4 = complex_mul(x_4, twiddle_4);
  x_5 = complex_mul(x_5, twiddle_5);
  x_6 = complex_mul(x_6, twiddle_6);

  float2 y_0 = x_0 + x_1 + x_2 + x_3 + x_4 + x_5 + x_6;
  float2 y_1 = x_0 + complex_mul(x_1, w_1) + complex_mul(x_2, w_2) + complex_mul(x_3, w_3) + complex_mul(x_4, w_4) + complex_mul(x_5, w_5) + complex_mul(x_6, w_6);
  float2 y_2 = x_0 + complex_mul(x_1, w_2) + complex_mul(x_2, w_4) + complex_mul(x_3, w_6) + complex_mul(x_4, w_1) + complex_mul(x_5, w_3) + complex_mul(x_6, w_5);
  float2 y_3 = x_0 + complex_mul(x_1, w_3) + complex_mul(x_2, w_6) + complex_mul(x_3, w_2) + complex_mul(x_4, w_5) + complex_mul(x_5, w_1) + complex_mul(x_6, w_4);
  float2 y_4 = x_0 + complex_mul(x_1, w_4) + complex_mul(x_2, w_1) + complex_mul(x_3, w_5) + complex_mul(x_4, w_2) + complex_mul(x_5, w_6) + complex_mul(x_6, w_3);
  float2 y_5 = x_0 + complex_mul(x_1, w_5) + complex_mul(x_2, w_3) + complex_mul(x_3, w_1) + complex_mul(x_4, w_6) + complex_mul(x_5, w_4) + complex_mul(x_6, w_2);
  float2 y_6 = x_0 + complex_mul(x_1, w_6) + complex_mul(x_2, w_5) + complex_mul(x_3, w_4) + complex_mul(x_4, w_3) + complex_mul(x_5, w_2) + complex_mul(x_6, w_1);

  write_buf[j] = y_0;
  write_buf[j + p] = y_1;
  write_buf[j + 2*p] = y_2;
  write_buf[j + 3*p] = y_3;
  write_buf[j + 4*p] = y_4;
  write_buf[j + 5*p] = y_5;
  write_buf[j + 6*p] = y_6;
}

void stockham_switch(threadgroup float2** read_buf, threadgroup float2** write_buf) {
    threadgroup float2* tmp = *write_buf;
    *write_buf = *read_buf;
    *read_buf = tmp;
}

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
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {

  int i = elem.z;
  int tg_idx = elem.y * n;
  int batch_idx = elem.x * grid.y * n + tg_idx;

  // The number of the threads we're using for each DFT
  int m = grid.z;

  // the thread group memory will be too big
  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = &shared_in[tg_idx];
  threadgroup float2* write_buf = &shared_out[tg_idx];

  // Account for possible extra threadgroups
  int grid_index = elem.x * grid.y + elem.y;
  if (grid_index >= batch_size) {
    return;
  }

  // Max registers is the max radix size supported
  int rader_m = 2;

  float2 x_0 = in[batch_idx + (i / rader_m)];

  // Basically when are we done?
  int p = 1;
  rader_n_step_forward<2, radix2>(i, p, m, rader_m, n, batch_idx, x_0, read_buf, in, out, raders_g_q, raders_b_q);
  
  p = 1;
  rader_n_step_backward<2, radix2>(i, p, m, rader_m, n, batch_idx, x_0, read_buf, out, raders_g_minus_q);

  p = 5;
  radix_n_step<2, radix2>(i, p, m, n, batch_idx, read_buf, in, out);
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

  int n_over_2 = (n/2) + 1;

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
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = &shared_in[tg_idx];
  threadgroup float2* write_buf = &shared_out[tg_idx];

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
      out[batch_idx_out + index + next_out] = complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
  // Add in elements up to n/2 + 1
  int num_left = n_over_2 - (elems_per_thread_ / 2 * m);
  if (fft_idx < num_left) {
    int index = fft_idx + elems_per_thread_ / 2 * m;
    float2 x_k = read_buf[index];
    float2 x_n_minus_k = read_buf[n - index] * conj;
    out[batch_idx_out + index] = (x_k + x_n_minus_k) / 2;
    out[batch_idx_out + index + next_out] = complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
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
  threadgroup float2 shared_out[tg_mem_size];
  threadgroup float2* read_buf = &shared_in[tg_idx];
  threadgroup float2* write_buf = &shared_out[tg_idx];

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
      float2 elem  = read_buf[index + length - 1] * inv_factor;
      elem = complex_mul(elem, w_k[index]);
      if (inv_) {
        elem *= inv_factor_overall;
      }
      out[batch_idx + index] = elem;
    }
  }
}


#define instantiate_fft(tg_mem_size) \
  template [[host_name("fft_mem_" #tg_mem_size)]] \
  [[kernel]] void fft<tg_mem_size>( \
      const device float2* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
      const device float2* raders_b_q [[buffer(2)]], \
      const device short* raders_g_q [[buffer(3)]], \
      const device short* raders_g_minus_q [[buffer(4)]], \
    constant const int& n, \
    constant const int& batch_size, \
    uint3 elem [[thread_position_in_grid]], \
    uint3 grid [[threads_per_grid]]);

#define instantiate_rfft(tg_mem_size) \
  template [[host_name("rfft_mem_" #tg_mem_size)]] \
  [[kernel]] void rfft<tg_mem_size>( \
      const device float* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
    constant const int& n, \
    constant const int& batch_size, \
    uint3 elem [[thread_position_in_grid]], \
    uint3 grid [[threads_per_grid]]);

#define instantiate_bluestein(tg_mem_size) \
  template [[host_name("bluestein_fft_mem_" #tg_mem_size)]] \
  [[kernel]] void bluestein_fft<tg_mem_size>( \
      const device float2* in [[buffer(0)]], \
      device float2* out [[buffer(1)]], \
      const device float2* w_q [[buffer(2)]], \
      const device float2* w_k [[buffer(2)]], \
    constant const int& length, \
    constant const int& n, \
    constant const int& batch_size, \
    uint3 elem [[thread_position_in_grid]], \
    uint3 grid [[threads_per_grid]]);

#define instantiate_ffts(tg_mem_size) \
    instantiate_fft(tg_mem_size) \
    instantiate_rfft(tg_mem_size) \
    instantiate_bluestein(tg_mem_size) \


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
// 2048 is the max that will fit into 32KB of threadgroup memory.
instantiate_ffts(2048)
