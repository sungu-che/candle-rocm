#include "hip_utils.h"
#include <math.h>

// Binary op: out[i] = lhs[lhs_i] OP rhs[rhs_i]
// info layout when non-null: [dims..., lhs_strides..., rhs_strides...]
#define BINARY_OP(NAME, EXPR)                                                    \
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

// Comparison op: output is u8 (0 or 1)
#define CMP_OP(NAME, EXPR)                                                       \
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

BINARY_OP(badd_f32, a + b)
BINARY_OP(bsub_f32, a - b)
BINARY_OP(bmul_f32, a * b)
BINARY_OP(bdiv_f32, a / b)
BINARY_OP(bmax_f32, fmaxf(a, b))
BINARY_OP(bmin_f32, fminf(a, b))

CMP_OP(eq_f32, a == b)
CMP_OP(ne_f32, a != b)
CMP_OP(lt_f32, a < b)
CMP_OP(le_f32, a <= b)
CMP_OP(gt_f32, a > b)
CMP_OP(ge_f32, a >= b)
