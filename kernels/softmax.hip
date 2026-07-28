#include "hip_utils.h"
#include <float.h>

__device__ inline float dev_expf(float x) { return __builtin_expf(x); }
__device__ inline float dev_logf(float x) { return __builtin_logf(x); }

// Numerically stable softmax: one block per row.
// Phase 1: find max (parallel reduce)
// Phase 2: compute sum of exp(x - max) (parallel reduce)
// Phase 3: out[i] = exp(x[i] - max) / sum
extern "C" __global__ void softmax_f32(
    const float *inp,
    float *out,
    const size_t nrows,
    const size_t ncols
) {
    extern __shared__ float shr[];
    const size_t row = blockIdx.x;
    if (row >= nrows) return;
    const size_t tid = threadIdx.x;
    const float *row_inp = inp + row * ncols;
    float *row_out = out + row * ncols;

    // Phase 1: find max
    float local_max = -FLT_MAX;
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        local_max = (local_max > row_inp[c] ? local_max : row_inp[c]);
    }
    shr[tid] = local_max;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] > shr[tid + s] ? shr[tid] : shr[tid + s]);
        __syncthreads();
    }
    float row_max = shr[0];
    __syncthreads();

    // Phase 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        local_sum += dev_expf(row_inp[c] - row_max);
    }
    shr[tid] = local_sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    float sum = shr[0];
    __syncthreads();

    // Phase 3: write normalized output
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        row_out[c] = dev_expf(row_inp[c] - row_max) / sum;
    }
}

// F16 softmax: reads F16, computes in F32 internally, outputs F16.
#include <hip/hip_fp16.h>

extern "C" __global__ void softmax_f16(
    const __half *inp,
    __half *out,
    const size_t nrows,
    const size_t ncols
) {
    extern __shared__ float shr[];
    const size_t row = blockIdx.x;
    if (row >= nrows) return;
    const size_t tid = threadIdx.x;
    const __half *row_inp = inp + row * ncols;
    __half *row_out = out + row * ncols;

    // Phase 1: find max (F32)
    float local_max = -FLT_MAX;
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        float v = __half2float(row_inp[c]);
        local_max = (local_max > v ? local_max : v);
    }
    shr[tid] = local_max;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] > shr[tid + s] ? shr[tid] : shr[tid + s]);
        __syncthreads();
    }
    float row_max = shr[0];
    __syncthreads();

    // Phase 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        local_sum += dev_expf(__half2float(row_inp[c]) - row_max);
    }
    shr[tid] = local_sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    float sum = shr[0];
    __syncthreads();

    // Phase 3: write normalized output (F16)
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        row_out[c] = __float2half(dev_expf(__half2float(row_inp[c]) - row_max) / sum);
    }
}

// log_softmax: out[i] = x[i] - max - log(sum(exp(x - max)))
extern "C" __global__ void log_softmax_f32(
    const float *inp,
    float *out,
    const size_t nrows,
    const size_t ncols
) {
    extern __shared__ float shr[];
    const size_t row = blockIdx.x;
    if (row >= nrows) return;
    const size_t tid = threadIdx.x;
    const float *row_inp = inp + row * ncols;
    float *row_out = out + row * ncols;

    // Phase 1: find max
    float local_max = -FLT_MAX;
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        local_max = (local_max > row_inp[c] ? local_max : row_inp[c]);
    }
    shr[tid] = local_max;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] > shr[tid + s] ? shr[tid] : shr[tid + s]);
        __syncthreads();
    }
    float row_max = shr[0];
    __syncthreads();

    // Phase 2: sum of exp(x - max)
    float local_sum = 0.0f;
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        local_sum += dev_expf(row_inp[c] - row_max);
    }
    shr[tid] = local_sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    float log_sum = dev_logf(shr[0]);
    __syncthreads();

    // Phase 3: out = x - max - log(sum)
    for (size_t c = tid; c < ncols; c += blockDim.x) {
        row_out[c] = row_inp[c] - row_max - log_sum;
    }
}