// Copyright © 2023 Apple Inc.
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <iostream>
#include <set>

namespace mlx::core {

using MTLFC = std::tuple<const void*, MTL::DataType, NS::UInteger>;

#define MAX_SINGLE_FFT_SIZE 2048
// Threadgroup memory batching improves throughput for small n
#define MIN_THREADGROUP_MEM_SIZE 64

int FFT::next_fast_n(int n) {
  return next_power_of_2(n);
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

struct FFTPlan {
  // Radix steps uses Stockham's algorithm normally
  std::vector<int> stockham;
  // Rader's algorithm decomposes prime - 1
  std::vector<int> rader;
};

std::vector<int> FFT::plan_stockham_fft(int n) {
  auto radices = FFT::supported_radices();
  std::vector<int> plan(radices.size(), 0);
  if (n == 1) {
    return plan;
  }
  for (int i = 0; i < radices.size(); i++) {
    int radix = radices[i];
    while (n % radix == 0) {
      plan[i] += 1;
      n /= radix;
      if (n == 1) {
        return plan;
      }
    }
  }
  // return an empty vector if unplannable
  throw std::runtime_error("Unplannable");
}

// Plan the sequence of radices
FFTPlan plan_fft(int n) {
  auto radices = FFT::supported_radices();
  std::set<int> radices_set(radices.begin(), radices.end());

  FFTPlan plan;
  plan.rader = std::vector<int>(radices.size(), 0);
  // A plan is a number of steps for each supported radix
  auto factors = prime_factors(n);
  for (int factor : factors) {
    // Make sure the factor is a supported radix
    if (radices_set.find(factor) == radices_set.end()) {
      // See if we can use Rader's algorithm to Stockham decompose n - 1
      auto rader_factors = prime_factors(factor - 1);
      int last_factor = -1;
      for (int rf : rader_factors) {
        // We don't nest Rader's algorithm so if one of the n - 1
        // factors isn't Stockham decomposable we give up and do Bluestein's
        if (radices_set.find(rf) == radices_set.end()) {
          return FFTPlan();
        }
      }
      plan.rader = FFT::plan_stockham_fft(factor - 1);
      n /= factor;
      break;
    }
  }

  // std::cout << "n " << std::endl;

  plan.stockham = FFT::plan_stockham_fft(n);
  return plan;
}

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& in = inputs[0];

  if (axes_.size() == 0) {
    throw std::runtime_error("GPU FFT is not implemented for 0D transforms.");
  }

  size_t n = out.dtype() == float32 ? out.shape(axes_[0]) : in.shape(axes_[0]);

  if (n > MAX_SINGLE_FFT_SIZE) {
    throw std::runtime_error(
        "fused GPU FFT is only implemented from 2 -> 2048");
  }

  // Make sure that the array is contiguous and has stride 1 in the FFT dim
  std::vector<array> copies;
  auto check_input = [this, &copies, &s](const array& x) {
    // TODO: Pass the strides to the kernel so
    // we can avoid the copy when x is not contiguous.
    bool no_copy = x.strides()[axes_[0]] == 1 && x.flags().row_contiguous ||
        x.flags().col_contiguous;
    if (no_copy) {
      return x;
    } else {
      array x_copy(x.shape(), x.dtype(), nullptr, {});
      std::vector<size_t> strides;
      size_t cur_stride = x.shape(axes_[0]);
      for (int axis = 0; axis < x.ndim(); axis++) {
        if (axis == axes_[0]) {
          strides.push_back(1);
        } else {
          strides.push_back(cur_stride);
          cur_stride *= x.shape(axis);
        }
      }

      auto flags = x.flags();
      size_t f_stride = 1;
      size_t b_stride = 1;
      flags.col_contiguous = true;
      flags.row_contiguous = true;
      for (int i = 0, ri = x.ndim() - 1; i < x.ndim(); ++i, --ri) {
        flags.col_contiguous &= (strides[i] == f_stride || x.shape(i) == 1);
        f_stride *= x.shape(i);
        flags.row_contiguous &= (strides[ri] == b_stride || x.shape(ri) == 1);
        b_stride *= x.shape(ri);
      }
      // This is probably over-conservative
      flags.contiguous = false;

      x_copy.set_data(
          allocator::malloc_or_wait(x.nbytes()), x.data_size(), strides, flags);
      copy_gpu_inplace(x, x_copy, CopyType::GeneralGeneral, s);
      copies.push_back(x_copy);
      return x_copy;
    }
  };
  const array& in_contiguous = check_input(inputs[0]);

  // real to complex: n -> (n/2)+1
  // complex to real: (n/2)+1 -> n
  auto out_strides = in_contiguous.strides();
  if (in.dtype() != out.dtype()) {
    for (int i = 0; i < out_strides.size(); i++) {
      if (out_strides[i] != 1) {
        out_strides[i] =
            out_strides[i] / in.shape(axes_[0]) * out.shape(axes_[0]);
      }
    }
  }
  // TODO: allow donation here
  out.set_data(
      allocator::malloc_or_wait(out.nbytes()),
      in_contiguous.data_size(),
      out_strides,
      {});

  int bluestein_n = -1;
  auto plan = plan_fft(n);
  auto radices = FFT::supported_radices();
  // for (int i = 0; i < 5; i++) {
  //   std::cout << "stockham plan " << radices[i] << " " << plan.stockham[i] <<
  //   std::endl; std::cout << "rader plan " << radices[i] << " " <<
  //   plan.rader[i] << std::endl;
  // }
  // if (plan.size() == 0) {
  //   // Bluestein's algorithm transforms an FFT to
  //   // a convolution of size > 2n + 1.
  //   // We solve that conv via FFT wth the convolution theorem.
  //   bluestein_n = next_fast_n(2 * n - 1);
  //   plan = plan_stockham_fft(bluestein_n);
  // }

  int fft_size = bluestein_n > 0 ? bluestein_n : n;

  // Setup function constants
  bool power_of_2 = is_power_of_2(fft_size);

  auto make_int = [](int* a, int i) {
    return std::make_tuple(a, MTL::DataType::DataTypeInt, i);
  };
  auto make_bool = [](bool* a, int i) {
    return std::make_tuple(a, MTL::DataType::DataTypeBool, i);
  };

  std::vector<MTLFC> func_consts = {
      make_bool(&inverse_, 0), make_bool(&power_of_2, 1)};

  // for (int p: plan.stockham) {
  //   std::cout << "stock p " << p << std::endl;
  // }
  // for (int p: plan.rader) {
  //   std::cout << "rader p " << p << std::endl;
  // }

  int index = 3;
  int elems_per_thread = 0;
  for (int i = 0; i < plan.stockham.size(); i++) {
    func_consts.push_back(make_int(&plan.stockham[i], index));
    index += 1;
    if (plan.stockham[i] > 0) {
      elems_per_thread = std::max(elems_per_thread, radices[i]);
    }
  }
  for (int i = 0; i < plan.rader.size(); i++) {
    func_consts.push_back(make_int(&plan.rader[i], index));
    index += 1;
    if (plan.rader[i] > 0) {
      elems_per_thread = std::max(elems_per_thread, radices[i]);
    }
  }
  func_consts.push_back(make_int(&elems_per_thread, 2));

  // The overall number of FFTs we're going to compute for this input
  int total_batch_size = in.size() / in.shape(axes_[0]);
  int threads_per_fft = fft_size / elems_per_thread;

  // We batch among threadgroups for improved efficiency when n is small
  int threadgroup_batch_size = std::max(MIN_THREADGROUP_MEM_SIZE / fft_size, 1);
  int threadgroup_mem_size = next_power_of_2(threadgroup_batch_size * fft_size);

  // ceil divide
  int batch_size =
      (total_batch_size + threadgroup_batch_size - 1) / threadgroup_batch_size;

  if (real_ && !inverse_) {
    // We can perform 2 RFFTs at once so the batch size is halved.
    batch_size = (batch_size + 2 - 1) / 2;
  }
  int out_buffer_size = out.size();

  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    std::ostringstream kname;
    std::string inv_string = inverse_ ? "true" : "false";
    if (bluestein_n > 0) {
      kname << "bluestein_fft_mem_" << threadgroup_mem_size;
    } else if (in.dtype() == float32) {
      kname << "rfft_mem_" << threadgroup_mem_size;
    } else {
      kname << "fft_mem_" << threadgroup_mem_size;
    }
    std::string base_name = kname.str();
    // We use a specialized kernel for each FFT size
    kname << "_n" << fft_size << "_inv_" << inverse_;
    std::string hash_name = kname.str();
    auto kernel = d.get_kernel(base_name, "mlx", hash_name, func_consts);

    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(in_contiguous, 0);
    compute_encoder.set_output_array(out, 1);

    auto& raders_b_q = inputs[1];
    auto& raders_g_q = inputs[2];
    auto& raders_g_minus_q = inputs[3];
    compute_encoder.set_input_array(raders_b_q, 2);
    compute_encoder.set_input_array(raders_g_q, 3);
    compute_encoder.set_input_array(raders_g_minus_q, 4);

    if (bluestein_n > 0) {
      // Precomputed twiddle factors for Bluestein's
      auto& w_q = inputs[1];
      auto& w_k = inputs[2];
      compute_encoder.set_input_array(w_q, 5); // w_q
      compute_encoder.set_input_array(w_k, 6); // w_k
      compute_encoder->setBytes(&n, sizeof(int), 7);
      compute_encoder->setBytes(&bluestein_n, sizeof(int), 8);
      compute_encoder->setBytes(&total_batch_size, sizeof(int), 9);
    } else {
      compute_encoder->setBytes(&n, sizeof(int), 5);
      compute_encoder->setBytes(&total_batch_size, sizeof(int), 6);
    }

    // std::cout << "fft size " << fft_size << std::endl;
    // std::cout << "threads per fft " << threads_per_fft << std::endl;
    // std::cout << "elems per thread " << elems_per_thread << std::endl;
    // std::cout << "threadgroup batch size " << threadgroup_batch_size <<
    // std::endl; std::cout << "batch size " << batch_size << std::endl;

    auto group_dims = MTL::Size(1, threadgroup_batch_size, threads_per_fft);
    auto grid_dims =
        MTL::Size(batch_size, threadgroup_batch_size, threads_per_fft);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

} // namespace mlx::core
