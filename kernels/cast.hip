#include "hip_utils.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Type casting kernels
#define CAST_OP(NAME, SRC_T, DST_T)                                              \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const SRC_T *inp,                                                            \
    DST_T *out                                                                   \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *strides = info ? info + num_dims : nullptr;                    \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        out[i] = (DST_T)inp[src_i];                                             \
    }                                                                            \
}

// f32 conversions (most common for picochat)
CAST_OP(cast_f32_u8, float, uint8_t)
CAST_OP(cast_f32_u32, float, uint32_t)
CAST_OP(cast_f32_i64, float, int64_t)
CAST_OP(cast_f32_f64, float, double)

CAST_OP(cast_u8_f32, uint8_t, float)
CAST_OP(cast_u32_f32, uint32_t, float)
CAST_OP(cast_i64_f32, int64_t, float)
CAST_OP(cast_f64_f32, double, float)

// f16 conversions (attention bottleneck — to_dtype F16↔F32)
CAST_OP(cast_f16_f32, __half, float)
CAST_OP(cast_f32_f16, float, __half)

// bf16 conversions (quantized layer norm / attention)
CAST_OP(cast_bf16_f32, __hip_bfloat16, float)
CAST_OP(cast_f32_bf16, float, __hip_bfloat16)

// BF16↔F16 conversions (for BF16 GEMM via hgemm on gfx1030)
// Must go through float to avoid ambiguous conversion.
extern "C" __global__ void cast_bf16_f16(
    const size_t numel, const size_t num_dims, const size_t *info,
    const __hip_bfloat16 *inp, __half *out
) {
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        size_t src_i = i;
        if (!is_contiguous(info)) {
            const size_t *dims = info;
            const size_t *strides = info + num_dims;
            src_i = get_strided_index(i, num_dims, dims, strides);
        }
        out[i] = __half(float(inp[src_i]));
    }
}
extern "C" __global__ void cast_f16_bf16(
    const size_t numel, const size_t num_dims, const size_t *info,
    const __half *inp, __hip_bfloat16 *out
) {
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {
        size_t src_i = i;
        if (!is_contiguous(info)) {
            const size_t *dims = info;
            const size_t *strides = info + num_dims;
            src_i = get_strided_index(i, num_dims, dims, strides);
        }
        out[i] = __hip_bfloat16(float(inp[src_i]));
    }
}

// u32/i64 conversions (used for index types)
CAST_OP(cast_u32_i64, uint32_t, int64_t)
CAST_OP(cast_i64_u32, int64_t, uint32_t)
CAST_OP(cast_u8_u32, uint8_t, uint32_t)
CAST_OP(cast_u32_u8, uint32_t, uint8_t)
