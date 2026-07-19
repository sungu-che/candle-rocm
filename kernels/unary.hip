#include "hip_utils.h"

// Unary op macro: out[i] = OP(inp[strided_i])
#define UNARY_OP(NAME, EXPR)                                                     \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const float *inp,                                                            \
    float *out                                                                   \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *strides = info ? info + num_dims : nullptr;                    \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        float x = inp[src_i];                                                    \
        out[i] = (EXPR);                                                         \
    }                                                                            \
}

// Unary op with one extra parameter
#define UNARY_OP1(NAME, EXPR)                                                    \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const size_t num_dims,                                                       \
    const size_t *info,                                                          \
    const float *inp,                                                            \
    float *out,                                                                  \
    const float param                                                            \
) {                                                                              \
    const size_t *dims = info;                                                   \
    const size_t *strides = info ? info + num_dims : nullptr;                    \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        float x = inp[src_i];                                                    \
        out[i] = (EXPR);                                                         \
    }                                                                            \
}

// Identity copy (used for contiguous-ification)
UNARY_OP(ucopy_f32, x)

// Standard math
UNARY_OP(uexp_f32, expf(x))
UNARY_OP(ulog_f32, logf(x))
UNARY_OP(usin_f32, sinf(x))
UNARY_OP(ucos_f32, cosf(x))
UNARY_OP(utanh_f32, tanhf(x))
UNARY_OP(usqrt_f32, sqrtf(x))
UNARY_OP(usqr_f32, x * x)
UNARY_OP(uneg_f32, -x)
UNARY_OP(urecip_f32, 1.0f / x)
UNARY_OP(uabs_f32, fabsf(x))
UNARY_OP(usign_f32, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))
UNARY_OP(ufloor_f32, floorf(x))
UNARY_OP(uceil_f32, ceilf(x))
UNARY_OP(uround_f32, roundf(x))

// Activations
UNARY_OP(urelu_f32, (x > 0.0f ? x : 0.0f))
UNARY_OP(usilu_f32, x / (1.0f + expf(-x)))
// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
UNARY_OP(ugelu_f32, 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))))
// GELU via erf: 0.5 * x * (1 + erf(x / sqrt(2)))
UNARY_OP(ugelu_erf_f32, 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)))
UNARY_OP(uerf_f32, erff(x))

// Parameterized ops
UNARY_OP1(upowf_f32, powf(x, param))
// ELU: x if x > 0, alpha * (exp(x) - 1) otherwise
UNARY_OP1(uelu_f32, (x > 0.0f) ? x : param * (expf(x) - 1.0f))
