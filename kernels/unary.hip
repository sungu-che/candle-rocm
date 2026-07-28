#include "hip_utils.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// Unary op macro: out[i] = OP(inp[strided_i]) — F32
#define UNARY_OP_F32(NAME, EXPR)                                                  \
extern "C" __global__ void NAME(                                                  \
    const size_t numel,                                                           \
    const size_t num_dims,                                                        \
    const size_t *info,                                                           \
    const float *inp,                                                             \
    float *out                                                                    \
) {                                                                               \
    const size_t *dims = info;                                                    \
    const size_t *strides = info ? info + num_dims : nullptr;                     \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                           \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        float x = inp[src_i];                                                     \
        out[i] = (EXPR);                                                          \
    }                                                                             \
}

// Unary op macro: out[i] = OP(inp[strided_i]) — F16
#define UNARY_OP_F16(NAME, EXPR)                                                  \
extern "C" __global__ void NAME(                                                  \
    const size_t numel,                                                           \
    const size_t num_dims,                                                        \
    const size_t *info,                                                           \
    const __half *inp,                                                            \
    __half *out                                                                   \
) {                                                                               \
    const size_t *dims = info;                                                    \
    const size_t *strides = info ? info + num_dims : nullptr;                     \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                           \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        float x = __half2float(inp[src_i]);                                        \
        out[i] = __float2half(EXPR);                                              \
    }                                                                             \
}

// Unary op macro: out[i] = OP(inp[strided_i]) — BF16
#define UNARY_OP_BF16(NAME, EXPR)                                                 \
extern "C" __global__ void NAME(                                                  \
    const size_t numel,                                                           \
    const size_t num_dims,                                                        \
    const size_t *info,                                                           \
    const __hip_bfloat16 *inp,                                                    \
    __hip_bfloat16 *out                                                           \
) {                                                                               \
    const size_t *dims = info;                                                    \
    const size_t *strides = info ? info + num_dims : nullptr;                     \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                           \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        float x = __bfloat162float(inp[src_i]);                                   \
        out[i] = __float2bfloat16(EXPR);                                          \
    }                                                                             \
}

// Unary op with one extra parameter — F32
#define UNARY_OP1_F32(NAME, EXPR)                                                 \
extern "C" __global__ void NAME(                                                  \
    const size_t numel,                                                           \
    const size_t num_dims,                                                        \
    const size_t *info,                                                           \
    const float *inp,                                                             \
    float *out,                                                                   \
    const float param                                                             \
) {                                                                               \
    const size_t *dims = info;                                                    \
    const size_t *strides = info ? info + num_dims : nullptr;                     \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                           \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        float x = inp[src_i];                                                     \
        out[i] = (EXPR);                                                          \
    }                                                                             \
}

// Unary op with one extra parameter — F16
#define UNARY_OP1_F16(NAME, EXPR)                                                 \
extern "C" __global__ void NAME(                                                  \
    const size_t numel,                                                           \
    const size_t num_dims,                                                        \
    const size_t *info,                                                           \
    const __half *inp,                                                            \
    __half *out,                                                                  \
    const float param                                                             \
) {                                                                               \
    const size_t *dims = info;                                                    \
    const size_t *strides = info ? info + num_dims : nullptr;                     \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                           \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        float x = __half2float(inp[src_i]);                                       \
        out[i] = __float2half(EXPR);                                              \
    }                                                                             \
}

// Unary op with one extra parameter — BF16
#define UNARY_OP1_BF16(NAME, EXPR)                                                \
extern "C" __global__ void NAME(                                                  \
    const size_t numel,                                                           \
    const size_t num_dims,                                                        \
    const size_t *info,                                                           \
    const __hip_bfloat16 *inp,                                                    \
    __hip_bfloat16 *out,                                                          \
    const float param                                                             \
) {                                                                               \
    const size_t *dims = info;                                                    \
    const size_t *strides = info ? info + num_dims : nullptr;                     \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                           \
        size_t src_i = is_contiguous(info) ? i : get_strided_index(i, num_dims, dims, strides); \
        float x = __bfloat162float(inp[src_i]);                                   \
        out[i] = __float2bfloat16(EXPR);                                          \
    }                                                                             \
}

// ============================================================
// F32 kernels
// ============================================================

// Identity copy (used for contiguous-ification)
UNARY_OP_F32(ucopy_f32, x)

// Standard math
UNARY_OP_F32(uexp_f32, expf(x))
UNARY_OP_F32(ulog_f32, logf(x))
UNARY_OP_F32(usin_f32, sinf(x))
UNARY_OP_F32(ucos_f32, cosf(x))
UNARY_OP_F32(utanh_f32, tanhf(x))
UNARY_OP_F32(usqrt_f32, sqrtf(x))
UNARY_OP_F32(usqr_f32, x * x)
UNARY_OP_F32(uneg_f32, -x)
UNARY_OP_F32(urecip_f32, 1.0f / x)
UNARY_OP_F32(uabs_f32, fabsf(x))
UNARY_OP_F32(usign_f32, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))
UNARY_OP_F32(ufloor_f32, floorf(x))
UNARY_OP_F32(uceil_f32, ceilf(x))
UNARY_OP_F32(uround_f32, roundf(x))

// Activations
UNARY_OP_F32(urelu_f32, (x > 0.0f ? x : 0.0f))
UNARY_OP_F32(usilu_f32, x / (1.0f + expf(-x)))
// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
UNARY_OP_F32(ugelu_f32, 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))))
// GELU via erf: 0.5 * x * (1 + erf(x / sqrt(2)))
UNARY_OP_F32(ugelu_erf_f32, 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)))
UNARY_OP_F32(uerf_f32, erff(x))

// Parameterized ops
UNARY_OP1_F32(upowf_f32, powf(x, param))
// ELU: x if x > 0, alpha * (exp(x) - 1) otherwise
UNARY_OP1_F32(uelu_f32, (x > 0.0f) ? x : param * (expf(x) - 1.0f))

// ============================================================
// F16 kernels
// ============================================================

UNARY_OP_F16(ucopy_f16, x)
UNARY_OP_F16(uexp_f16, expf(x))
UNARY_OP_F16(ulog_f16, logf(x))
UNARY_OP_F16(usin_f16, sinf(x))
UNARY_OP_F16(ucos_f16, cosf(x))
UNARY_OP_F16(utanh_f16, tanhf(x))
UNARY_OP_F16(usqrt_f16, sqrtf(x))
UNARY_OP_F16(usqr_f16, x * x)
UNARY_OP_F16(uneg_f16, -x)
UNARY_OP_F16(urecip_f16, 1.0f / x)
UNARY_OP_F16(uabs_f16, fabsf(x))
UNARY_OP_F16(usign_f16, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))
UNARY_OP_F16(ufloor_f16, floorf(x))
UNARY_OP_F16(uceil_f16, ceilf(x))
UNARY_OP_F16(uround_f16, roundf(x))

// Activations
UNARY_OP_F16(urelu_f16, (x > 0.0f ? x : 0.0f))
UNARY_OP_F16(usilu_f16, x / (1.0f + expf(-x)))
UNARY_OP_F16(ugelu_f16, 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))))
UNARY_OP_F16(ugelu_erf_f16, 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)))
UNARY_OP_F16(uerf_f16, erff(x))

// Parameterized ops
UNARY_OP1_F16(upowf_f16, powf(x, param))
UNARY_OP1_F16(uelu_f16, (x > 0.0f) ? x : param * (expf(x) - 1.0f))

// ============================================================
// BF16 kernels
// ============================================================

UNARY_OP_BF16(ucopy_bf16, x)
UNARY_OP_BF16(uexp_bf16, expf(x))
UNARY_OP_BF16(ulog_bf16, logf(x))
UNARY_OP_BF16(usin_bf16, sinf(x))
UNARY_OP_BF16(ucos_bf16, cosf(x))
UNARY_OP_BF16(utanh_bf16, tanhf(x))
UNARY_OP_BF16(usqrt_bf16, sqrtf(x))
UNARY_OP_BF16(usqr_bf16, x * x)
UNARY_OP_BF16(uneg_bf16, -x)
UNARY_OP_BF16(urecip_bf16, 1.0f / x)
UNARY_OP_BF16(uabs_bf16, fabsf(x))
UNARY_OP_BF16(usign_bf16, (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f))
UNARY_OP_BF16(ufloor_bf16, floorf(x))
UNARY_OP_BF16(uceil_bf16, ceilf(x))
UNARY_OP_BF16(uround_bf16, roundf(x))

// Activations
UNARY_OP_BF16(urelu_bf16, (x > 0.0f ? x : 0.0f))
UNARY_OP_BF16(usilu_bf16, x / (1.0f + expf(-x)))
UNARY_OP_BF16(ugelu_bf16, 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))))
UNARY_OP_BF16(ugelu_erf_bf16, 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)))
UNARY_OP_BF16(uerf_bf16, erff(x))

// Parameterized ops
UNARY_OP1_BF16(upowf_bf16, powf(x, param))
UNARY_OP1_BF16(uelu_bf16, (x > 0.0f) ? x : param * (expf(x) - 1.0f))