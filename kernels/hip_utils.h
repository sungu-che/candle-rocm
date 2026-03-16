#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

// Strided index: converts flat thread index to memory offset for non-contiguous tensors.
// dims_and_strides layout: [dim0, dim1, ..., stride0, stride1, ...]
__device__ __forceinline__ size_t get_strided_index(
    size_t idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t strided_i = 0;
    for (int d = (int)num_dims - 1; d >= 0; d--) {
        strided_i += (idx % dims[d]) * strides[d];
        idx /= dims[d];
    }
    return strided_i;
}

// Check if dims_and_strides is null (signals contiguous layout from Rust side)
__device__ __forceinline__ bool is_contiguous(const size_t *info) {
    return info == nullptr;
}

// Grid-stride loop bounds
#define GS_IDX (blockIdx.x * blockDim.x + threadIdx.x)
#define GS_STEP (blockDim.x * gridDim.x)

// Launch config helper: 256 threads/block, enough blocks to cover numel
#define LAUNCH_CFG(numel) dim3((((numel) + 255) / 256)), dim3(256)
