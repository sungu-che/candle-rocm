// Fused GroupNorm kernel for UNet attention blocks.
// Replaces 5-8 CPU fallback ops with a single GPU kernel.
//
// GroupNorm(x, gamma, beta) = gamma * (x - mean) / sqrt(var + eps) + beta
// where mean/var are computed per group over (C/G, H, W)

#include "hip_utils.h"

// One block per (batch, group). Each block normalizes C_per_group * H * W elements.
extern "C" __global__ void group_norm_f32(
    const float* __restrict__ input,   // (N, C, H, W)
    const float* __restrict__ gamma,   // (C,)
    const float* __restrict__ beta,    // (C,)
    float* __restrict__ output,        // (N, C, H, W)
    int N, int C, int H, int W,
    int num_groups,
    float eps
) {
    extern __shared__ float shr[];
    
    int group_idx = blockIdx.x;  // 0..N*num_groups-1
    int n = group_idx / num_groups;
    int g = group_idx % num_groups;
    if (n >= N) return;
    
    int C_per_group = C / num_groups;
    int spatial = H * W;
    int group_size = C_per_group * spatial;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Pass 1: compute mean
    float sum = 0.0f;
    for (int i = tid; i < group_size; i += block_size) {
        int c_local = i / spatial;
        int sw = i % spatial;
        int c = g * C_per_group + c_local;
        int idx = ((n * C + c) * spatial) + sw;
        sum += input[idx];
    }
    shr[tid] = sum;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    float mean = shr[0] / (float)group_size;
    __syncthreads();
    
    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < group_size; i += block_size) {
        int c_local = i / spatial;
        int sw = i % spatial;
        int c = g * C_per_group + c_local;
        int idx = ((n * C + c) * spatial) + sw;
        float diff = input[idx] - mean;
        var_sum += diff * diff;
    }
    shr[tid] = var_sum;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shr[tid] += shr[tid + s];
        __syncthreads();
    }
    float inv_std = 1.0f / sqrtf(shr[0] / (float)group_size + eps);
    
    // Pass 3: normalize + affine
    for (int i = tid; i < group_size; i += block_size) {
        int c_local = i / spatial;
        int sw = i % spatial;
        int c = g * C_per_group + c_local;
        int idx = ((n * C + c) * spatial) + sw;
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}
