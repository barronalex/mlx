// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

enum class CopyType {
  // Copy a raw scalar input into the full contiguous output
  Scalar,

  // Copy the raw input buffer contiguously into a raw output buffer of the same
  // size
  Vector,

  // Copy the full virtual input to the full contiguous output
  General,

  // Copy the full virtual input to the full virtual output. We assume the
  // input and output have the same shape.
  GeneralGeneral
};

void copy(const array& src, array& dst, CopyType ctype);
void copy_inplace(const array& src, array& dst, CopyType ctype);

template <typename stride_t>
void copy_inplace(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    const std::vector<stride_t>& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype);

std::tuple<bool, int64_t, std::vector<int64_t>> prepare_slice(
    const array& in,
    std::vector<int>& start_indices,
    std::vector<int>& strides);

void shared_buffer_slice(
    const array& in,
    const std::vector<size_t>& out_strides,
    size_t data_offset,
    array& out);

void slice_op(
    const array& in,
    array& out,
    std::vector<int>& start_indices,
    std::vector<int>& strides);

void concatenate_op(const std::vector<array>& inputs, array out, int axis);

} // namespace mlx::core
