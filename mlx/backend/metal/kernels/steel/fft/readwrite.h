#include <metal_common>

#include "mlx/backend/metal/kernels/steel/utils.h"
/* FFT helpers for reading and writing from/to device memory.

For many sizes, GPU FFTs are memory bandwidth bound so
read/write performance is crucial.

Where possible, we read 128 bits sequentially in each thread,
coalesced with accesses from adajcent threads for optimal performance.
*/

#define MAX_RADIX 13

using namespace metal;

template <typename T>
struct ReadWriter {
  const device T* in;
  threadgroup float2* buf;
  device float2* out;
  int n;
  int batch_size;
  int elems_per_thread;
  uint3 elem;
  uint3 grid;
  int threads_per_tg;

  METAL_FUNC ReadWriter(
      const device T* in_,
      threadgroup float2* buf_,
      device float2* out_,
      const short n_,
      const int batch_size_,
      const short elems_per_thread_,
      const uint3 elem_,
      const uint3 grid_)
      : in(in_),
        buf(buf_),
        out(out_),
        n(n_),
        batch_size(batch_size_),
        elems_per_thread(elems_per_thread_),
        elem(elem_),
        grid(grid_) {
    // Account for padding on last threadgroup
    threads_per_tg = elem.x == grid.x - 1
        ? ((batch_size / 2) - (grid.x - 1) * grid.y) * grid.z
        : grid.y * grid.z;
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
      buf[index] = in[batch_idx + index];
      buf[index + 1] = in[batch_idx + index + 1];
    }
    if (elems_per_thread % 2 != 0) {
      short index = tg_idx +
          read_width * threads_per_tg * (elems_per_thread / read_width);
      buf[index] = in[batch_idx + index];
    }
  }

  METAL_FUNC void write() const {
    int batch_idx = elem.x * grid.y * n;
    short tg_idx = elem.y * grid.z + elem.z;

    constexpr int read_width = 2;
    for (short e = 0; e < (elems_per_thread / read_width); e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      // vectorized reads
      out[batch_idx + index] = buf[index];
      out[batch_idx + index + 1] = buf[index + 1];
    }
    if (elems_per_thread % 2 != 0) {
      short index = tg_idx +
          read_width * threads_per_tg * (elems_per_thread / read_width);
      out[batch_idx + index] = buf[index];
    }
  }

  METAL_FUNC void load_strided(int stride) const {
    // int coalesce_width = grid.y;
    // int tg_idx = elem.y * grid.z + elem.z;
    // int shared_idx = (tg_idx % coalesce_width) * n +
    //     tg_idx / coalesce_width * elems_per_thread;
    // int outer_batch_size = (stride / coalesce_width);
    // int base_batch_idx = (elem.x % outer_batch_size) * coalesce_width +
    //     overall_n * (elem.x / outer_batch_size);
    // int device_idx = base_batch_idx +
    //     tg_idx / coalesce_width * elems_per_thread * stride +
    //     tg_idx % coalesce_width;

    // for (int e = 0; e < elems_per_thread; e++) {
    //   buf[shared_idx + e] = in[device_idx + e * stride];
    // }
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
bool ReadWriter<float>::out_of_bounds() const {
  int grid_index = elem.x * grid.y + elem.y;
  // We pack two sequences into one for RFFTs
  return grid_index * 2 >= batch_size;
}

template <>
void ReadWriter<float>::load() const {
  int batch_idx = elem.x * grid.y * n * 2;
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
void ReadWriter<float>::write() const {
  short n_over_2 = (n / 2) + 1;

  int batch_idx = elem.x * grid.y * n_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n_over_2;

  float2 conj = {1, -1};
  float2 minus_j = {0, -1};

  short m = grid.z;
  short fft_idx = elem.z;

  for (int t = 0; t < elems_per_thread / 2; t++) {
    int index = metal::min(fft_idx + t * m, n_over_2 - 1);
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
  // Add in elements up to n/2 + 1
  int num_left = n_over_2 - (elems_per_thread / 2 * m);
  if (fft_idx < num_left) {
    int index = metal::min(fft_idx + elems_per_thread / 2 * m, n_over_2 - 1);
    float2 x_k = seq_buf[index];
    float2 x_n_minus_k = seq_buf[n - index] * conj;
    out[batch_idx + index] = (x_k + x_n_minus_k) / 2;
    out[batch_idx + index + next_out] =
        complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
  }
}

// Strided loading

// Padded loading