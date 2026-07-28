#include "hip_utils.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Mixed BF16×F32 binary ops: BF16 lhs, F32 rhs → BF16 out.
// Avoids CPU fallback for RoPE mul, attention bias add, mask mul etc.
// Kernel does: out[i] = bf16(op(float(lhs[i]), rhs[i]))

#define BINARY_MIXED_OP(NAME, OP_EXPR)                                          \
extern "C" __global__ void NAME(                                                \
    const size_t numel,                                                         \
    const size_t num_dims,                                                      \
    const size_t *info,                                                         \
    const __hip_bfloat16 *lhs,                                                  \
    const float *rhs,                                                           \
    __hip_bfloat16 *out                                                         \
) {                                                                             \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                         \
        size_t lhs_i = i;                                                       \
        size_t rhs_i = i;                                                       \
        if (!is_contiguous(info)) {                                             \
            const size_t *dims = info;                                          \
            const size_t *strides = info + num_dims;                            \
            lhs_i = get_strided_index(i, num_dims, dims, strides);             \
        }                                                                       \
        float a = float(lhs[lhs_i]);                                            \
        float b = rhs[rhs_i];                                                   \
        out[i] = __hip_bfloat16(OP_EXPR);                                       \
    }                                                                           \
}

// Element-wise ops (no broadcast — both same shape)
BINARY_MIXED_OP(mul_bf16_f32, a * b)
BINARY_MIXED_OP(add_bf16_f32, a + b)
BINARY_MIXED_OP(sub_bf16_f32, a - b)

// Scalar broadcast: rhs is a single F32 value, broadcast to all elements.
#define BINARY_MIXED_SCALAR_OP(NAME, OP_EXPR)                                   \
extern "C" __global__ void NAME(                                                \
    const size_t numel,                                                         \
    const __hip_bfloat16 *lhs,                                                  \
    const float *rhs_scalar,                                                    \
    __hip_bfloat16 *out                                                         \
) {                                                                             \
    float b = rhs_scalar[0];                                                    \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                         \
        float a = float(lhs[i]);                                                \
        out[i] = __hip_bfloat16(OP_EXPR);                                       \
    }                                                                           \
}

BINARY_MIXED_SCALAR_OP(mul_bf16_f32_scalar, a * b)
BINARY_MIXED_SCALAR_OP(add_bf16_f32_scalar, a + b)