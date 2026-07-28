#include "hip_utils.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <float.h>

// =====================================================================
// F32 REDUCTION KERNELS
// fast_sum_f32: sum reduce
// fast_max_f32: max reduce
// fast_min_f32: min reduce
// fast_argmax_f32: argmax (value + index in shared mem)
// fast_argmin_f32: argmin (value + index in shared mem)
// =====================================================================

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
        val = (val > inp[src_i] ? val : inp[src_i]);
    }
    shr[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] > shr[tid + s] ? shr[tid] : shr[tid + s]);
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
        val = (val < inp[src_i] ? val : inp[src_i]);
    }
    shr[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] < shr[tid + s] ? shr[tid] : shr[tid + s]);
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

// =====================================================================
// F16 REDUCTION KERNELS (accumulate in float for precision)
// =====================================================================

extern "C" __global__ void fast_sum_f16(
    const __half *inp,
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
        sum += __half2float(inp[src_i]);
    }
    shr[tid] = sum;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

extern "C" __global__ void fast_max_f16(
    const __half *inp,
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
        float v = __half2float(inp[src_i]);
        val = (val > v ? val : v);
    }
    shr[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] > shr[tid + s] ? shr[tid] : shr[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

extern "C" __global__ void fast_min_f16(
    const __half *inp,
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
        float v = __half2float(inp[src_i]);
        val = (val < v ? val : v);
    }
    shr[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] < shr[tid + s] ? shr[tid] : shr[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

extern "C" __global__ void fast_argmax_f16(
    const __half *inp,
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
        float v = __half2float(inp[src_i]);
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

extern "C" __global__ void fast_argmin_f16(
    const __half *inp,
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
        float v = __half2float(inp[src_i]);
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

// =====================================================================
// BF16 REDUCTION KERNELS (accumulate in float for precision)
// =====================================================================

extern "C" __global__ void fast_sum_bf16(
    const __hip_bfloat16 *inp,
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
        sum += __bfloat162float(inp[src_i]);
    }
    shr[tid] = sum;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

extern "C" __global__ void fast_max_bf16(
    const __hip_bfloat16 *inp,
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
        float v = __bfloat162float(inp[src_i]);
        val = (val > v ? val : v);
    }
    shr[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] > shr[tid + s] ? shr[tid] : shr[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

extern "C" __global__ void fast_min_bf16(
    const __hip_bfloat16 *inp,
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
        float v = __bfloat162float(inp[src_i]);
        val = (val < v ? val : v);
    }
    shr[tid] = val;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] = (shr[tid] < shr[tid + s] ? shr[tid] : shr[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[out_idx] = shr[0];
}

extern "C" __global__ void fast_argmax_bf16(
    const __hip_bfloat16 *inp,
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
        float v = __bfloat162float(inp[src_i]);
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

extern "C" __global__ void fast_argmin_bf16(
    const __hip_bfloat16 *inp,
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
        float v = __bfloat162float(inp[src_i]);
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