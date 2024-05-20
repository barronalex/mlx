#include <metal_common>

#include "mlx/backend/metal/kernels/steel/utils.h"
/* FFT helpers for reading and writing from/to device memory.

For many sizes, GPU FFTs are memory bandwidth bound so
read/write performance is crucial.

Where possible, we read 128 bits sequentially in each thread,
coalesced with accesses from adajcent threads for optimal performance.
*/

/* Here are all the cases we need to support:

Reading:
  - Contiguous
    - Complex
      - Radix
      - Rader
      - Bluestein
    - Real
      - Radix
      - Rader
      - Bluestein
  - Strided
    - Complex
      - Radix
      - Rader
      - Bluestein
    - Real
      - Radix
      - Rader
      - Bluestein
*/

template <typename T>
struct ReadWriter {
  // For maximal throughput each
  // thread reads 128 bits at a time
  // (2 complex64s)
  // x x - - x x -
  // - x x - - x x
  // - - x x - - x
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

  // Contiguous memory case
  METAL_FUNC void load() const {
    // Index of the batch
    int batch_idx = elem.x * grid.y * n;

    // Index within threadgroup
    short tg_idx = elem.y * grid.z + elem.z;

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

  // Contiguous memory case
  METAL_FUNC void write() const {
    // Index of the batch
    int batch_idx = elem.x * grid.y * n;

    // Index within threadgroup
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
  short tg_idx = elem.y * grid.z + elem.z;
  // Read 4 float32s sequentially per thread to reach 128 bits
  constexpr int read_width = 4;

  short num_elems = (elems_per_thread / (read_width / 2));
  threadgroup float* float_buf = (threadgroup float*)buf;
  for (short e = 0; e < num_elems; e++) {
    // TODO: move within the i loop
    short index = read_width * tg_idx + read_width * threads_per_tg * e;

    // Two real sequences are packed into one complex one:
    // x x x x - - - - o o o o + + + +  ->  x - x - x - x - o + o + o + o +
    short even_seq = (index / n) % 2;
    short buf_index = (index % n) * 2 + even_seq + (index / (2 * n)) * 2 * n;

    // vectorized reads
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < read_width; i++) {
      // Pack two float32 seqs into one complex64 seq.
      float_buf[buf_index + 2 * i] = in[batch_idx + index + i];
    }
  }
  // if (elems_per_thread % 2 != 0) {
  //   short index = (read_width / 2) * tg_idx +
  //       read_width * threads_per_tg * num_elems;
  //   short seq = (index / n) % 2;
  //   buf[index][seq] = in[batch_idx + index];
  //   seq = ((index + 1) / n) % 2;
  //   buf[index + 1][seq] = in[batch_idx + index + 1];
  // }
}

template <>
void ReadWriter<float>::write() const {
  int n_over_2 = (n / 2) + 1;
  int batch_idx_out = elem.x * grid.y * n_over_2 * 2;
  short tg_idx = elem.y * grid.z + elem.z;

  constexpr int read_width = 2;
  short num_elems = elems_per_thread / 2;
  for (short e = 0; e < num_elems; e++) {
    short index = read_width * tg_idx + read_width * threads_per_tg * e;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < read_width; i++) {
      out[batch_idx_out + index + i] = buf[index + i];
    }
  }

  // auto conj = float2(1, -1);
  // auto minus_j = float2(0, -1);

  // for (short e = 0; e < num_elems; e++) {
  //   short index = read_width * (tg_idx + threads_per_tg * e);

  //   float2 temp[read_width];
  //   STEEL_PRAGMA_UNROLL
  //   for (short i = 0; i < read_width; i++) {
  //     short n_index = (index + i) % n_over_2;
  //     short even_seq = ((index + i) / n_over_2) % 2;
  //     short start_index = (index / (n_over_2 * 2)) * n;

  //     float2 x_k = buf[start_index + n_index];
  //     float2 x_n_minus_k = buf[start_index + n - n_index] * conj;
  //     if (n_index == 0) {
  //       temp[i] = float2(x_k[even_seq], 0);
  //     } else if (even_seq == 0) {
  //       temp[i] = (x_k + x_n_minus_k) / 2;
  //     } else {
  //       temp[i] = complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
  //     }
  //     // temp[i] = (x_k + x_n_minus_k) / 2;
  //   }

  //   // Vectorized writes
  //   STEEL_PRAGMA_UNROLL
  //   for (short i = 0; i < read_width; i++) {
  //     out[batch_idx_out + index + i] = temp[i];
  //   }
  // }

  // // Add on one more for the "+1" in "(n / 2) + 1"
  // // TODO: only need to do this when output is odd length
  // short index = tg_idx + read_width * threads_per_tg * num_elems;
  // short n_index = index % n_over_2;
  // short even_seq = (index / n_over_2) % 2;
  // short start_index = (index / (n_over_2 * 2)) * n;

  // float2 x_k = buf[start_index + n_index];
  // float2 x_n_minus_k = buf[start_index + n - n_index] * conj;
  // float2 output;
  // if (n_index == 0) {
  //   output = float2(x_k[even_seq], 0);
  // } else if (even_seq == 0) {
  //   output = (x_k + x_n_minus_k) / 2;
  // } else {
  //   output = complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
  // }
  // // output = (x_k + x_n_minus_k) / 2;
  // out[batch_idx_out + index] = output;
}

// Strided loading

// Padded loading