// Fused RMSNorm kernel: y = x * rsqrt(mean(x^2) + eps) * weight
// One block per row. No shared memory needed for small dims (warp reduce).
// For large dims (>blockDim), uses grid-stride loop + shared memory reduce.

#include "hip_utils.h"

// Device-side math helpers using __builtin (avoid glibc math.h conflict)
__device__ inline float dev_rsqrtf(float x) { return 1.0f / __builtin_sqrtf(x); }
__device__ inline float dev_sqrtf(float x) { return __builtin_sqrtf(x); }

extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ input,   // (n_rows, n_cols)
    float* __restrict__ output,        // (n_rows, n_cols)
    const float* __restrict__ weight,  // (n_cols,)
    const int n_cols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_in = input + row * n_cols;
    float* row_out = output + row * n_cols;

    // Phase 1: compute sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        float v = row_in[i];
        local_sum += v * v;
    }

    // Warp-level reduce (RDNA2 wavefront = 32)
    // Use shared memory for cross-warp reduction
    __shared__ float shared[32]; // max 32 warps per block (1024 threads / 32)

    // Warp reduce via shuffle
    float val = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Final reduce across warps
    int num_warps = (block_size + 31) / 32;
    float sum_sq = 0.0f;
    if (tid < num_warps) {
        sum_sq = shared[tid];
    }
    // Simple single-warp reduce for final stage
    if (tid == 0) {
        for (int i = 1; i < num_warps; i++) {
            sum_sq += shared[i];
        }
        shared[0] = dev_rsqrtf(sum_sq / (float)n_cols + eps);
    }
    __syncthreads();

    float inv_std = shared[0];

    // Phase 2: normalize and scale
    for (int i = tid; i < n_cols; i += block_size) {
        row_out[i] = row_in[i] * inv_std * weight[i];
    }
}

// Fused BF16 RMSNorm kernel: y = x * rsqrt(mean(x^2) + eps) * weight
// Input/output in bfloat16, weight in float32, computation in float32.
// One block per row. Reduces to F32 for the sum, then normalizes.
#include <hip/hip_bf16.h>

extern "C" __global__ void rmsnorm_bf16(
    const __hip_bfloat16* __restrict__ input,   // (n_rows, n_cols)
    __hip_bfloat16* __restrict__ output,        // (n_rows, n_cols)
    const float* __restrict__ weight,          // (n_cols,) F32
    const int n_cols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __hip_bfloat16* row_in = input + row * n_cols;
    __hip_bfloat16* row_out = output + row * n_cols;

    // Phase 1: compute sum of squares in F32
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        float v = __bfloat162float(row_in[i]);
        local_sum += v * v;
    }

    // Warp-level reduce
    __shared__ float shared[32];
    float val = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < num_warps; i++) sum_sq += shared[i];
        shared[0] = dev_rsqrtf(sum_sq / (float)n_cols + eps);
    }
    __syncthreads();

    float inv_std = shared[0];

    // Phase 2: normalize and scale, cast back to BF16
    for (int i = tid; i < n_cols; i += block_size) {
        float normalized = __bfloat162float(row_in[i]) * inv_std * weight[i];
        row_out[i] = __float2bfloat16(normalized);
    }
}

// Fused BF16×BF16 RMSNorm kernel: y = x * rsqrt(mean(x^2) + eps) * weight
// Input/output/weight all bfloat16. Computation in float32.
// One block per row. Reduces to F32 for the sum, then normalizes.
extern "C" __global__ void rmsnorm_bf16_bf16(
    const __hip_bfloat16* __restrict__ input,   // (n_rows, n_cols)
    __hip_bfloat16* __restrict__ output,        // (n_rows, n_cols)
    const __hip_bfloat16* __restrict__ weight,   // (n_cols,) BF16
    const int n_cols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __hip_bfloat16* row_in = input + row * n_cols;
    __hip_bfloat16* row_out = output + row * n_cols;

    // Phase 1: compute sum of squares in F32
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        float v = __bfloat162float(row_in[i]);
        local_sum += v * v;
    }

    // Warp-level reduce
    __shared__ float shared[32];
    float val = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < num_warps; i++) sum_sq += shared[i];
        shared[0] = dev_rsqrtf(sum_sq / (float)n_cols + eps);
    }
    __syncthreads();

    float inv_std = shared[0];

    // Phase 2: normalize and scale, cast back to BF16
    for (int i = tid; i < n_cols; i += block_size) {
        float normalized = __bfloat162float(row_in[i]) * inv_std * __bfloat162float(weight[i]);
        row_out[i] = __float2bfloat16(normalized);
    }
}

// Fused F16×F16 RMSNorm kernel: y = x * rsqrt(mean(x^2) + eps) * weight
// Input/output/weight all __half. Computation in float32.
#include <hip/hip_fp16.h>

extern "C" __global__ void rmsnorm_f16_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const __half* __restrict__ weight,
    const int n_cols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* row_in = input + row * n_cols;
    __half* row_out = output + row * n_cols;

    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        float v = __half2float(row_in[i]);
        local_sum += v * v;
    }

    __shared__ float shared[32];
    float val = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < num_warps; i++) sum_sq += shared[i];
        shared[0] = dev_rsqrtf(sum_sq / (float)n_cols + eps);
    }
    __syncthreads();

    float inv_std = shared[0];

    for (int i = tid; i < n_cols; i += block_size) {
        float normalized = __half2float(row_in[i]) * inv_std * __half2float(weight[i]);
        row_out[i] = __float2half(normalized);
    }
}

// Fused F16 RMSNorm kernel: y = x * rsqrt(mean(x^2) + eps) * weight
// Uses __half from hip_fp16.h (already included above)

extern "C" __global__ void rmsnorm_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const float* __restrict__ weight,
    const int n_cols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* row_in = input + row * n_cols;
    __half* row_out = output + row * n_cols;

    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        float v = __half2float(row_in[i]);
        local_sum += v * v;
    }

    __shared__ float shared[32];
    float val = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < num_warps; i++) sum_sq += shared[i];
        shared[0] = dev_rsqrtf(sum_sq / (float)n_cols + eps);
    }
    __syncthreads();

    float inv_std = shared[0];

    for (int i = tid; i < n_cols; i += block_size) {
        float normalized = __half2float(row_in[i]) * inv_std * weight[i];
        row_out[i] = __float2half(normalized);
    }
}

// Fused LayerNorm kernel: y = (x - mean) / sqrt(var + eps) * weight + bias
// One block per row. Two-pass: mean, then normalize.
extern "C" __global__ void layernorm_f32(
    const float* __restrict__ input,   // (n_rows, n_cols)
    float* __restrict__ output,        // (n_rows, n_cols)
    const float* __restrict__ weight,  // (n_cols,)
    const float* __restrict__ bias,    // (n_cols,)
    const int n_cols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_in = input + row * n_cols;
    float* row_out = output + row * n_cols;

    __shared__ float shared[32]; // max 32 warps

    // Phase 1: compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        local_sum += row_in[i];
    }

    // Warp reduce
    float val = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    float mean = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < num_warps; i++) mean += shared[i];
        mean /= (float)n_cols;
        shared[0] = mean;
    }
    __syncthreads();
    mean = shared[0];

    // Phase 2: compute variance
    float local_var = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }

    val = local_var;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    float inv_std = 0.0f;
    if (tid == 0) {
        float var_sum = 0.0f;
        for (int i = 0; i < num_warps; i++) var_sum += shared[i];
        inv_std = dev_rsqrtf(var_sum / (float)n_cols + eps);
        shared[0] = inv_std;
    }
    __syncthreads();
    inv_std = shared[0];

    // Phase 3: normalize + affine
    for (int i = tid; i < n_cols; i += block_size) {
        float normalized = (row_in[i] - mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// LayerNorm without bias: y = (x - mean) / sqrt(var + eps) * weight
extern "C" __global__ void layernorm_no_bias_f32(
    const float* __restrict__ input,   // (n_rows, n_cols)
    float* __restrict__ output,        // (n_rows, n_cols)
    const float* __restrict__ weight,  // (n_cols,)
    const int n_cols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_in = input + row * n_cols;
    float* row_out = output + row * n_cols;

    __shared__ float shared[32];

    // Phase 1: compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        local_sum += row_in[i];
    }

    float val = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    float mean = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < num_warps; i++) mean += shared[i];
        mean /= (float)n_cols;
        shared[0] = mean;
    }
    __syncthreads();
    mean = shared[0];

    // Phase 2: compute variance
    float local_var = 0.0f;
    for (int i = tid; i < n_cols; i += block_size) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }

    val = local_var;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    float inv_std = 0.0f;
    if (tid == 0) {
        float var_sum = 0.0f;
        for (int i = 0; i < num_warps; i++) var_sum += shared[i];
        inv_std = dev_rsqrtf(var_sum / (float)n_cols + eps);
        shared[0] = inv_std;
    }
    __syncthreads();
    inv_std = shared[0];

    // Phase 3: normalize * weight
    for (int i = tid; i < n_cols; i += block_size) {
        row_out[i] = (row_in[i] - mean) * inv_std * weight[i];
    }
}
