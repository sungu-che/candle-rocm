#include "hip_utils.h"
#include <math.h>
#include <float.h>

// Reduction kernels: one block per output element.
// Each block reduces `reduce_size` input elements using shared memory.
//
// Params:
//   inp:         input data
//   out:         output (one per block)
//   info:        [dims..., strides...] of the full input tensor
//   out_numel:   number of output elements
//   reduce_size: number of elements to reduce per output
//   num_dims:    rank of input tensor

extern "C" __global__ void fast_sum_f32(
    const float *inp,
    float *out,
    const size_t *info,
    const size_t out_numel,
    const size_t reduce_size,
    const size_t num_dims
) {
    extern __shared__ float shr[];
    const size_t tid = threadIdx.x;
    const size_t out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;

    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    float sum = 0.0f;
    for (size_t r = tid; r < reduce_size; r += blockDim.x) {
        size_t flat = out_idx * reduce_size + r;
        size_t src_i = get_strided_index(flat, num_dims, dims, strides);
        sum += inp[src_i];
    }
    shr[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

extern "C" __global__ void fast_max_f32(
    const float *inp,
    float *out,
    const size_t *info,
    const size_t out_numel,
    const size_t reduce_size,
    const size_t num_dims
) {
    extern __shared__ float shr[];
    const size_t tid = threadIdx.x;
    const size_t out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;

    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    float val = -FLT_MAX;
    for (size_t r = tid; r < reduce_size; r += blockDim.x) {
        size_t flat = out_idx * reduce_size + r;
        size_t src_i = get_strided_index(flat, num_dims, dims, strides);
        val = fmaxf(val, inp[src_i]);
    }
    shr[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = fmaxf(shr[tid], shr[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

extern "C" __global__ void fast_min_f32(
    const float *inp,
    float *out,
    const size_t *info,
    const size_t out_numel,
    const size_t reduce_size,
    const size_t num_dims
) {
    extern __shared__ float shr[];
    const size_t tid = threadIdx.x;
    const size_t out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;

    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    float val = FLT_MAX;
    for (size_t r = tid; r < reduce_size; r += blockDim.x) {
        size_t flat = out_idx * reduce_size + r;
        size_t src_i = get_strided_index(flat, num_dims, dims, strides);
        val = fminf(val, inp[src_i]);
    }
    shr[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = fminf(shr[tid], shr[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

// Argmax: track both value and index in shared memory.
// Shared layout: float[blockDim.x] values, then uint32_t[blockDim.x] indices.
extern "C" __global__ void fast_argmax_f32(
    const float *inp,
    uint32_t *out,
    const size_t *info,
    const size_t out_numel,
    const size_t reduce_size,
    const size_t num_dims
) {
    extern __shared__ char shr_raw[];
    float *shr_val = (float *)shr_raw;
    uint32_t *shr_idx = (uint32_t *)(shr_raw + blockDim.x * sizeof(float));

    const size_t tid = threadIdx.x;
    const size_t out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;

    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    float best_val = -FLT_MAX;
    uint32_t best_idx = 0;
    for (size_t r = tid; r < reduce_size; r += blockDim.x) {
        size_t flat = out_idx * reduce_size + r;
        size_t src_i = get_strided_index(flat, num_dims, dims, strides);
        float v = inp[src_i];
        if (v > best_val) { best_val = v; best_idx = (uint32_t)r; }
    }
    shr_val[tid] = best_val;
    shr_idx[tid] = best_idx;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shr_val[tid + s] > shr_val[tid]) {
                shr_val[tid] = shr_val[tid + s];
                shr_idx[tid] = shr_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr_idx[0];
}

extern "C" __global__ void fast_argmin_f32(
    const float *inp,
    uint32_t *out,
    const size_t *info,
    const size_t out_numel,
    const size_t reduce_size,
    const size_t num_dims
) {
    extern __shared__ char shr_raw[];
    float *shr_val = (float *)shr_raw;
    uint32_t *shr_idx = (uint32_t *)(shr_raw + blockDim.x * sizeof(float));

    const size_t tid = threadIdx.x;
    const size_t out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;

    const size_t *dims = info;
    const size_t *strides = info + num_dims;

    float best_val = FLT_MAX;
    uint32_t best_idx = 0;
    for (size_t r = tid; r < reduce_size; r += blockDim.x) {
        size_t flat = out_idx * reduce_size + r;
        size_t src_i = get_strided_index(flat, num_dims, dims, strides);
        float v = inp[src_i];
        if (v < best_val) { best_val = v; best_idx = (uint32_t)r; }
    }
    shr_val[tid] = best_val;
    shr_idx[tid] = best_idx;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shr_val[tid + s] < shr_val[tid]) {
                shr_val[tid] = shr_val[tid + s];
                shr_idx[tid] = shr_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr_idx[0];
}
