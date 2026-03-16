#include "hip_utils.h"

// Fill buffer with constant value
extern "C" __global__ void fill_f32(
    float *out,
    const size_t numel,
    const float val
) {
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        out[i] = val;
    }
}

extern "C" __global__ void fill_u8(
    uint8_t *out,
    const size_t numel,
    const uint8_t val
) {
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        out[i] = val;
    }
}

extern "C" __global__ void fill_u32(
    uint32_t *out,
    const size_t numel,
    const uint32_t val
) {
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        out[i] = val;
    }
}

// Copy strided source to contiguous destination
// src may be strided (described by info), dst is always contiguous
extern "C" __global__ void copy_strided_f32(
    const float *src,
    float *dst,
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const size_t src_offset,
    const size_t dst_offset
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        size_t src_i = get_strided_index(i, num_dims, dims, strides) + src_offset;
        dst[i + dst_offset] = src[src_i];
    }
}

// 2D copy with different src/dst strides (like cudaMemcpy2D)
extern "C" __global__ void copy2d_f32(
    const float *src,
    float *dst,
    const size_t d1,
    const size_t d2,
    const size_t src_stride1,
    const size_t dst_stride1,
    const size_t src_offset,
    const size_t dst_offset
) {
    size_t numel = d1 * d2;
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        size_t row = i / d2;
        size_t col = i % d2;
        dst[dst_offset + row * dst_stride1 + col] = src[src_offset + row * src_stride1 + col];
    }
}
