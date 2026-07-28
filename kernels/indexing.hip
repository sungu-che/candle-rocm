#include "hip_utils.h"
#include <hip/hip_bf16.h>

// Indexing operations follow candle's decomposition:
// The dimension being indexed is split into left_size * dim_size * right_size.
// This avoids dealing with arbitrary-rank tensors in the kernel.

// gather: out[left, ids_idx, right] = inp[left, ids[left, ids_idx, right], right]
#define GATHER_OP(IDX_T, NAME)                                                   \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const IDX_T *ids,                                                            \
    const float *inp,                                                            \
    float *out,                                                                  \
    const size_t left_size,                                                      \
    const size_t src_dim_size,                                                   \
    const size_t ids_dim_size,                                                   \
    const size_t right_size                                                      \
) {                                                                              \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t right = i % right_size;                                           \
        size_t ids_idx = (i / right_size) % ids_dim_size;                        \
        size_t left = i / (right_size * ids_dim_size);                           \
        IDX_T idx = ids[left * ids_dim_size * right_size + ids_idx * right_size + right]; \
        size_t src_i = left * src_dim_size * right_size + (size_t)idx * right_size + right; \
        out[i] = inp[src_i];                                                     \
    }                                                                            \
}

GATHER_OP(uint32_t, gather_u32_f32)
GATHER_OP(int64_t, gather_i64_f32)

// index_select: out[left, ids_idx, right] = inp[left, ids[ids_idx], right]
#define INDEX_SELECT_OP(IDX_T, NAME)                                             \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const IDX_T *ids,                                                            \
    const float *inp,                                                            \
    float *out,                                                                  \
    const size_t left_size,                                                      \
    const size_t src_dim_size,                                                   \
    const size_t ids_dim_size,                                                   \
    const size_t right_size                                                      \
) {                                                                              \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t right = i % right_size;                                           \
        size_t ids_idx = (i / right_size) % ids_dim_size;                        \
        size_t left = i / (right_size * ids_dim_size);                           \
        IDX_T idx = ids[ids_idx];                                                \
        size_t src_i = left * src_dim_size * right_size + (size_t)idx * right_size + right; \
        out[i] = inp[src_i];                                                     \
    }                                                                            \
}

INDEX_SELECT_OP(uint32_t, index_select_u32_f32)
INDEX_SELECT_OP(int64_t, index_select_i64_f32)

// index_select BF16 variants
#define INDEX_SELECT_BF16_OP(IDX_T, NAME)                                         \
extern "C" __global__ void NAME(                                                   \
    const size_t numel,                                                            \
    const IDX_T *ids,                                                              \
    const __hip_bfloat16 *inp,                                                     \
    __hip_bfloat16 *out,                                                           \
    const size_t left_size,                                                        \
    const size_t src_dim_size,                                                     \
    const size_t ids_dim_size,                                                     \
    const size_t right_size                                                        \
) {                                                                                \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                            \
        size_t right = i % right_size;                                             \
        size_t ids_idx = (i / right_size) % ids_dim_size;                          \
        size_t left = i / (right_size * ids_dim_size);                             \
        IDX_T idx = ids[ids_idx];                                                  \
        size_t src_i = left * src_dim_size * right_size + (size_t)idx * right_size + right; \
        out[i] = inp[src_i];                                                       \
    }                                                                              \
}

INDEX_SELECT_BF16_OP(uint32_t, index_select_bf16)
INDEX_SELECT_BF16_OP(int64_t, index_select_i64_bf16)

// index_select F16 variants (same kernel, __half type)
#include <hip/hip_fp16.h>
#define INDEX_SELECT_F16_OP(IDX_T, NAME)                                         \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const IDX_T *ids,                                                            \
    const __half *inp,                                                           \
    __half *out,                                                                 \
    const size_t left_size,                                                      \
    const size_t src_dim_size,                                                   \
    const size_t ids_dim_size,                                                   \
    const size_t right_size                                                       \
) {                                                                              \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t right = i % right_size;                                           \
        size_t ids_idx = (i / right_size) % ids_dim_size;                        \
        size_t left = i / (right_size * ids_dim_size);                           \
        IDX_T idx = ids[ids_idx];                                                \
        size_t src_i = left * src_dim_size * right_size + (size_t)idx * right_size + right; \
        out[i] = inp[src_i];                                                     \
    }                                                                            \
}

INDEX_SELECT_F16_OP(uint32_t, index_select_f16)
INDEX_SELECT_F16_OP(int64_t, index_select_i64_f16)

// index_add: dst[left, ids[ids_idx], right] += src[left, ids_idx, right]
#define INDEX_ADD_OP(IDX_T, NAME)                                                \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const IDX_T *ids,                                                            \
    const size_t ids_dim_size,                                                   \
    const float *src,                                                            \
    float *dst,                                                                  \
    const size_t left_size,                                                      \
    const size_t dst_dim_size,                                                   \
    const size_t right_size                                                      \
) {                                                                              \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t right = i % right_size;                                           \
        size_t ids_idx = (i / right_size) % ids_dim_size;                        \
        size_t left = i / (right_size * ids_dim_size);                           \
        IDX_T idx = ids[ids_idx];                                                \
        size_t dst_i = left * dst_dim_size * right_size + (size_t)idx * right_size + right; \
        atomicAdd(&dst[dst_i], src[i]);                                          \
    }                                                                            \
}

INDEX_ADD_OP(uint32_t, index_add_u32_f32)
INDEX_ADD_OP(int64_t, index_add_i64_f32)

// scatter_add: dst[left, ids[left, idx, right], right] += src[left, idx, right]
#define SCATTER_ADD_OP(IDX_T, NAME)                                              \
extern "C" __global__ void NAME(                                                 \
    const size_t numel,                                                          \
    const IDX_T *ids,                                                            \
    const float *src,                                                            \
    float *dst,                                                                  \
    const size_t left_size,                                                      \
    const size_t src_dim_size,                                                   \
    const size_t dst_dim_size,                                                   \
    const size_t right_size                                                      \
) {                                                                              \
    for (size_t i = GS_IDX; i < numel; i += GS_STEP) {                          \
        size_t right = i % right_size;                                           \
        size_t src_idx = (i / right_size) % src_dim_size;                        \
        size_t left = i / (right_size * src_dim_size);                           \
        IDX_T idx = ids[left * src_dim_size * right_size + src_idx * right_size + right]; \
        size_t dst_i = left * dst_dim_size * right_size + (size_t)idx * right_size + right; \
        atomicAdd(&dst[dst_i], src[i]);                                          \
    }                                                                            \
}

SCATTER_ADD_OP(uint32_t, scatter_add_u32_f32)
SCATTER_ADD_OP(int64_t, scatter_add_i64_f32)
