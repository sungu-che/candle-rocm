#include "hip_utils.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Binary op: out[i] = lhs[lhs_i] OP rhs[rhs_i]  (all F32)
#define BINARY_OP_F32(NAME, EXPR)                                                \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const float *lhs,                                                            \
    const float *rhs,                                                            \
    float *out                                                                   \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *lhs_strides = info ? info + num_dims : nullptr;                \
    const size_t *rhs_strides = info ? info + 2 * num_dims : nullptr;            \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t lhs_i, rhs_i;                                                     \
        if (is_contiguous(info)) {                                               \
            lhs_i = i; rhs_i = i;                                                \
        } else {                                                                 \
            lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);           \
            rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);           \
        }                                                                        \
        float a = lhs[lhs_i];                                                    \
        float b = rhs[rhs_i];                                                    \
        out[i] = (EXPR);                                                         \
    }                                                                            \
}

// Binary op: all F16
#define BINARY_OP_F16(NAME, EXPR)                                                \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const __half *lhs,                                                           \
    const __half *rhs,                                                           \
    __half *out                                                                  \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *lhs_strides = info ? info + num_dims : nullptr;                \
    const size_t *rhs_strides = info ? info + 2 * num_dims : nullptr;            \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t lhs_i, rhs_i;                                                     \
        if (is_contiguous(info)) {                                               \
            lhs_i = i; rhs_i = i;                                                \
        } else {                                                                 \
            lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);           \
            rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);           \
        }                                                                        \
        __half a = lhs[lhs_i];                                                   \
        __half b = rhs[rhs_i];                                                   \
        out[i] = (EXPR);                                                         \
    }                                                                            \
}

// Binary op: all BF16
#define BINARY_OP_BF16(NAME, EXPR)                                               \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const __hip_bfloat16 *lhs,                                                   \
    const __hip_bfloat16 *rhs,                                                   \
    __hip_bfloat16 *out                                                          \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *lhs_strides = info ? info + num_dims : nullptr;                \
    const size_t *rhs_strides = info ? info + 2 * num_dims : nullptr;            \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t lhs_i, rhs_i;                                                     \
        if (is_contiguous(info)) {                                               \
            lhs_i = i; rhs_i = i;                                                \
        } else {                                                                 \
            lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);           \
            rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);           \
        }                                                                        \
        __hip_bfloat16 a = lhs[lhs_i];                                           \
        __hip_bfloat16 b = rhs[rhs_i];                                           \
        out[i] = (EXPR);                                                         \
    }                                                                            \
}

// Comparison op: output is u8 (0 or 1), input F32
#define CMP_OP_F32(NAME, EXPR)                                                   \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const float *lhs,                                                            \
    const float *rhs,                                                            \
    uint8_t *out                                                                 \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *lhs_strides = info ? info + num_dims : nullptr;                \
    const size_t *rhs_strides = info ? info + 2 * num_dims : nullptr;            \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t lhs_i, rhs_i;                                                     \
        if (is_contiguous(info)) {                                               \
            lhs_i = i; rhs_i = i;                                                \
        } else {                                                                 \
            lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);           \
            rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);           \
        }                                                                        \
        float a = lhs[lhs_i];                                                    \
        float b = rhs[rhs_i];                                                    \
        out[i] = (uint8_t)(EXPR);                                                \
    }                                                                            \
}

// Comparison op: output is u8, input F16
#define CMP_OP_F16(NAME, EXPR)                                                   \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const __half *lhs,                                                           \
    const __half *rhs,                                                           \
    uint8_t *out                                                                 \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *lhs_strides = info ? info + num_dims : nullptr;                \
    const size_t *rhs_strides = info ? info + 2 * num_dims : nullptr;            \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t lhs_i, rhs_i;                                                     \
        if (is_contiguous(info)) {                                               \
            lhs_i = i; rhs_i = i;                                                \
        } else {                                                                 \
            lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);           \
            rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);           \
        }                                                                        \
        float a = __half2float(lhs[lhs_i]);                                      \
        float b = __half2float(rhs[rhs_i]);                                      \
        out[i] = (uint8_t)(EXPR);                                                \
    }                                                                            \
}

// Comparison op: output is u8, input BF16
#define CMP_OP_BF16(NAME, EXPR)                                                  \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const __hip_bfloat16 *lhs,                                                   \
    const __hip_bfloat16 *rhs,                                                   \
    uint8_t *out                                                                 \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *lhs_strides = info ? info + num_dims : nullptr;                \
    const size_t *rhs_strides = info ? info + 2 * num_dims : nullptr;            \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t lhs_i, rhs_i;                                                     \
        if (is_contiguous(info)) {                                               \
            lhs_i = i; rhs_i = i;                                                \
        } else {                                                                 \
            lhs_i = get_strided_index(i, num_dims, dims, lhs_strides);           \
            rhs_i = get_strided_index(i, num_dims, dims, rhs_strides);           \
        }                                                                        \
        float a = __bfloat162float(lhs[lhs_i]);                                   \
        float b = __bfloat162float(rhs[rhs_i]);                                   \
        out[i] = (uint8_t)(EXPR);                                                \
    }                                                                            \
}

// F32 binary ops
BINARY_OP_F32(badd_f32, a + b)
BINARY_OP_F32(bsub_f32, a - b)
BINARY_OP_F32(bmul_f32, a * b)
BINARY_OP_F32(bdiv_f32, a / b)
BINARY_OP_F32(bmaximum_f32, (a > b ? a : b))
BINARY_OP_F32(bminimum_f32, (a < b ? a : b))

// F16 binary ops
BINARY_OP_F16(badd_f16, a + b)
BINARY_OP_F16(bsub_f16, a - b)
BINARY_OP_F16(bmul_f16, a * b)
BINARY_OP_F16(bdiv_f16, a / b)
BINARY_OP_F16(bmaximum_f16, (a > b ? a : b))
BINARY_OP_F16(bminimum_f16, (a < b ? a : b))

// BF16 binary ops
BINARY_OP_BF16(badd_bf16, a + b)
BINARY_OP_BF16(bsub_bf16, a - b)
BINARY_OP_BF16(bmul_bf16, a * b)
BINARY_OP_BF16(bdiv_bf16, a / b)
BINARY_OP_BF16(bmaximum_bf16, (a > b ? a : b))
BINARY_OP_BF16(bminimum_bf16, (a < b ? a : b))

// F32 comparison ops → u8
CMP_OP_F32(eq_f32, a == b)
CMP_OP_F32(ne_f32, a != b)
CMP_OP_F32(lt_f32, a < b)
CMP_OP_F32(le_f32, a <= b)
CMP_OP_F32(gt_f32, a > b)
CMP_OP_F32(ge_f32, a >= b)

// F16 comparison ops → u8
CMP_OP_F16(eq_f16, a == b)
CMP_OP_F16(ne_f16, a != b)
CMP_OP_F16(lt_f16, a < b)
CMP_OP_F16(le_f16, a <= b)
CMP_OP_F16(gt_f16, a > b)
CMP_OP_F16(ge_f16, a >= b)

// BF16 comparison ops → u8
CMP_OP_BF16(eq_bf16, a == b)
CMP_OP_BF16(ne_bf16, a != b)
CMP_OP_BF16(lt_bf16, a < b)
CMP_OP_BF16(le_bf16, a <= b)
CMP_OP_BF16(gt_bf16, a > b)
CMP_OP_BF16(ge_bf16, a >= b)