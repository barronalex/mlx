#include <metal_common>

#include "mlx/backend/metal/kernels/steel/utils.h"
/* FFT helpers for reading and writing from/to device memory.

For many sizes, GPU FFTs are memory bandwidth bound so
read/write performance is important.

Where possible, we read 128 bits sequentially in each thread,
coalesced with accesses from adajcent threads for optimal performance.
*/

#define MAX_RADIX 13

using namespace metal;

template <typename in_T, typename out_T>
struct ReadWriter {
  const device in_T* in;
  threadgroup float2* buf;
  device out_T* out;
  int n;
  int batch_size;
  int elems_per_thread;
  uint3 elem;
  uint3 grid;
  int threads_per_tg;
  bool inv;

  METAL_FUNC ReadWriter(
      const device in_T* in_,
      threadgroup float2* buf_,
      device out_T* out_,
      const short n_,
      const int batch_size_,
      const short elems_per_thread_,
      const uint3 elem_,
      const uint3 grid_,
      const bool inv_)
      : in(in_),
        buf(buf_),
        out(out_),
        n(n_),
        batch_size(batch_size_),
        elems_per_thread(elems_per_thread_),
        elem(elem_),
        grid(grid_),
        inv(inv_) {
    // Account for padding on last threadgroup
    threads_per_tg = elem.x == grid.x - 1
        ? (batch_size - (grid.x - 1) * grid.y) * grid.z
        : grid.y * grid.z;
  }

  // ifft(x) = 1/n * conj(fft(conj(x)))
  METAL_FUNC float2 post_in(float2 elem) const {
    return inv ? float2(elem.x, -elem.y) : elem;
  }

  METAL_FUNC float2 pre_out(float2 elem) const {
    return inv ? float2(elem.x / n, -elem.y / n) : elem;
  }

  METAL_FUNC float2 pre_out(float2 elem, int length) const {
    return inv ? float2(elem.x / length, -elem.y / length) : elem;
  }

  METAL_FUNC bool out_of_bounds() const {
    // Account for possible extra threadgroups
    int grid_index = elem.x * grid.y + elem.y;
    return grid_index >= batch_size;
  }

  METAL_FUNC void load() const {
    int batch_idx = elem.x * grid.y * n;
    short tg_idx = elem.y * grid.z + elem.z;

    // 2 complex64s = 128 bits
    constexpr int read_width = 2;
    for (short e = 0; e < (elems_per_thread / read_width); e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      // vectorized reads
      buf[index] = post_in(in[batch_idx + index]);
      buf[index + 1] = post_in(in[batch_idx + index + 1]);
    }
    if (elems_per_thread % 2 != 0) {
      short index = tg_idx +
          read_width * threads_per_tg * (elems_per_thread / read_width);
      buf[index] = post_in(in[batch_idx + index]);
    }
  }

  METAL_FUNC void write() const {
    int batch_idx = elem.x * grid.y * n;
    short tg_idx = elem.y * grid.z + elem.z;

    constexpr int read_width = 2;
    for (short e = 0; e < (elems_per_thread / read_width); e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      // vectorized reads
      out[batch_idx + index] = pre_out(buf[index]);
      out[batch_idx + index + 1] = pre_out(buf[index + 1]);
    }
    if (elems_per_thread % 2 != 0) {
      short index = tg_idx +
          read_width * threads_per_tg * (elems_per_thread / read_width);
      out[batch_idx + index] = pre_out(buf[index]);
    }
  }

  METAL_FUNC void load_padded(int length, const device float2* w_k) const {
    // Padded load for Bluestein's algorithm
    int batch_idx = elem.x * grid.y * length + elem.y * length;
    int fft_idx = elem.z;
    int m = grid.z;

    threadgroup float2* seq_buf = buf + elem.y * n;
    for (int e = 0; e < elems_per_thread; e++) {
      int index = fft_idx + e * m;
      if (index < length) {
        float2 elem = post_in(in[batch_idx + index]);
        seq_buf[index] = complex_mul(elem, w_k[index]);
      } else {
        seq_buf[index] = 0.0;
      }
    }
  }

  METAL_FUNC void write_padded(int length, const device float2* w_k) const {
    // Padded write for Bluestein's algorithm
    int batch_idx = elem.x * grid.y * length + elem.y * length;
    int fft_idx = elem.z;
    int m = grid.z;
    float2 inv_factor = {1.0f / n, -1.0f / n};

    threadgroup float2* seq_buf = buf + elem.y * n;
    for (int e = 0; e < elems_per_thread; e++) {
      int index = fft_idx + e * m;
      if (index < length) {
        float2 elem = seq_buf[index + length - 1] * inv_factor;
        out[batch_idx + index] = pre_out(complex_mul(elem, w_k[index]), length);
      }
    }
  }
};

// For RFFT, we interleave batches of two real sequences into one complex one:
//
// z_k = x_k + j.y_k
// X_k = (Z_k + Z_(N-k)*) / 2
// Y_k = -j * ((Z_k - Z_(N-k)*) / 2)
//
// This roughly doubles the throughput over the regular FFT.
template <>
bool ReadWriter<float, float2>::out_of_bounds() const {
  int grid_index = elem.x * grid.y + elem.y;
  // We pack two sequences into one for RFFTs
  return grid_index * 2 >= batch_size;
}

template <>
void ReadWriter<float, float2>::load() const {
  int batch_idx = elem.x * grid.y * n * 2 + elem.y * n * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n;

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    seq_buf[index].x = in[batch_idx + index];
    seq_buf[index].y = in[batch_idx + index + next_in];
  }
}

template <>
void ReadWriter<float, float2>::write() const {
  short n_over_2 = (n / 2) + 1;

  int batch_idx = elem.x * grid.y * n_over_2 * 2 + elem.y * n_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n_over_2;

  float2 conj = {1, -1};
  float2 minus_j = {0, -1};

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread / 2 + 1; e++) {
    int index = metal::min(fft_idx + e * m, n_over_2 - 1);
    // x_0 = z_0.real
    // y_0 = z_0.imag
    if (index == 0) {
      out[batch_idx + index] = {seq_buf[index].x, 0};
      out[batch_idx + index + next_out] = {seq_buf[index].y, 0};
    } else {
      float2 x_k = seq_buf[index];
      float2 x_n_minus_k = seq_buf[n - index] * conj;
      out[batch_idx + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx + index + next_out] =
          complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
}

// For IRFFT, we do the opposite
//
// Z_k = X_k + j.Y_k
// x_k = Re(Z_k)
// Y_k = Imag(Z_k)
template <>
bool ReadWriter<float2, float>::out_of_bounds() const {
  int grid_index = elem.x * grid.y + elem.y;
  // We pack two sequences into one for IRFFTs
  return grid_index * 2 >= batch_size;
}

template <>
void ReadWriter<float2, float>::load() const {
  short n_over_2 = (n / 2) + 1;
  int batch_idx = elem.x * grid.y * n_over_2 * 2 + elem.y * n_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n_over_2;

  short m = grid.z;
  short fft_idx = elem.z;

  float2 conj = {1, -1};
  float2 plus_j = {0, 1};

  for (int t = 0; t < elems_per_thread / 2 + 1; t++) {
    int index = metal::min(fft_idx + t * m, n_over_2 - 1);
    float2 x = in[batch_idx + index];
    float2 y = in[batch_idx + index + next_in];
    bool last_val = n % 2 == 0 && index == n_over_2 - 1;
    // NumPy ensures last input to even irfft is real
    if (last_val) {
      x = float2(x.x, 0);
      y = float2(y.x, 0);
    }
    seq_buf[index] = x + complex_mul(y, plus_j);
    seq_buf[index].y = -seq_buf[index].y;
    if (index > 0 && !last_val) {
      seq_buf[n - index] = (x * conj) + complex_mul(y * conj, plus_j);
      seq_buf[n - index].y = -seq_buf[n - index].y;
    }
  }
}

template <>
void ReadWriter<float2, float>::write() const {
  int batch_idx = elem.x * grid.y * n * 2 + elem.y * n * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n;

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread; e++) {
    int index = fft_idx + e * m;
    out[batch_idx + index] = seq_buf[index].x / n;
    out[batch_idx + index + next_out] = seq_buf[index].y / -n;
  }
}

template <typename in_T, typename out_T>
struct StridedReadWriter {
  const device in_T* in;
  threadgroup float2* buf;
  device out_T* out;
  int n;
  int batch_size;
  int elems_per_thread;
  uint3 elem;
  uint3 grid;
  int threads_per_tg;
  bool inv;
  int device_idx;
  int shared_idx;

  METAL_FUNC StridedReadWriter(
      const device in_T* in_,
      threadgroup float2* buf_,
      device out_T* out_,
      const short n_,
      const int batch_size_,
      const short elems_per_thread_,
      const uint3 elem_,
      const uint3 grid_,
      const bool inv_)
      : in(in_),
        buf(buf_),
        out(out_),
        n(n_),
        batch_size(batch_size_),
        elems_per_thread(elems_per_thread_),
        elem(elem_),
        grid(grid_),
        inv(inv_) {}
};