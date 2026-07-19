// im2col + bias_add kernels for GPU conv2d
// Used by candle-rocm's conv2d via im2col + rocBLAS GEMM

#include "hip_utils.h"
#include <hip/hip_fp16.h>

// im2col: (N, C, H, W) -> (N, C*Kh*Kw, H_out*W_out)
// Each thread: one (n, h_out, w_out) -> harvest C*Kh*Kw input values
extern "C" __global__ void im2col_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K_h, int K_w,
    int pad, int stride, int dilation,
    int H_out, int W_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out;
    if (tid >= total) return;

    int w_out = tid % W_out;
    int h_out = (tid / W_out) % H_out;
    int n = tid / (H_out * W_out);

    int h_start = h_out * stride - pad;
    int w_start = w_out * stride - pad;

    int col_stride = H_out * W_out;  // stride in output for each row

    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < K_h; kh++) {
            for (int kw = 0; kw < K_w; kw++) {
                int h_in = h_start + kh * dilation;
                int w_in = w_start + kw * dilation;
                int row = c * K_h * K_w + kh * K_w + kw;
                int col_idx = row * col_stride + (h_out * W_out + w_out);

                float val = 0.0f;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int inp_idx = ((n * C + c) * H + h_in) * W + w_in;
                    val = input[inp_idx];
                }
                output[col_idx] = val;
            }
        }
    }
}

// F16 version: same logic, half precision
extern "C" __global__ void im2col_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int C, int H, int W,
    int K_h, int K_w,
    int pad, int stride, int dilation,
    int H_out, int W_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out;
    if (tid >= total) return;

    int w_out = tid % W_out;
    int h_out = (tid / W_out) % H_out;
    int n = tid / (H_out * W_out);

    int h_start = h_out * stride - pad;
    int w_start = w_out * stride - pad;

    int col_stride = H_out * W_out;

    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < K_h; kh++) {
            for (int kw = 0; kw < K_w; kw++) {
                int h_in = h_start + kh * dilation;
                int w_in = w_start + kw * dilation;
                int row = c * K_h * K_w + kh * K_w + kw;
                int col_idx = row * col_stride + (h_out * W_out + w_out);

                __half val = __float2half(0.0f);
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int inp_idx = ((n * C + c) * H + h_in) * W + w_in;
                    val = input[inp_idx];
                }
                output[col_idx] = val;
            }
        }
    }
}

// bias_add: C_out vector broadcast over (1, C_out, H, W) tensor
extern "C" __global__ void bias_add_f32(
    float* __restrict__ tensor,
    const float* __restrict__ bias,
    int C_out, int spatial
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * spatial;
    if (tid >= total) return;
    int c = tid / spatial;
    tensor[tid] += bias[c];
}

extern "C" __global__ void bias_add_f16(
    __half* __restrict__ tensor,
    const __half* __restrict__ bias,
    int C_out, int spatial
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * spatial;
    if (tid >= total) return;
    int c = tid / spatial;
    tensor[tid] = __hadd(tensor[tid], bias[c]);
}