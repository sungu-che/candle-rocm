#include "hip_utils.h"
#include <hip/hip_fp16.h>

// Pool2D kernels: avg_pool2d + max_pool2d, F32 + F16
// Each thread computes one output element.

// ─── avg_pool2d ──────────────────────────────────────────────────────────────

extern "C" __global__ void avg_pool2d_f32(
    const float* __input,
    float*       __output,
    int b, int c, int ih, int iw,
    int kh, int kw,
    int pad_h, int pad_w,
    int str_h, int str_w,
    int oh, int ow
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * c * oh * ow;
    if (gid >= total) return;

    int idx = gid;
    int b_idx = idx / (c * oh * ow); idx %= (c * oh * ow);
    int c_idx = idx / (oh * ow);     idx %= (oh * ow);
    int oy    = idx / ow;
    int ox    = idx % ow;

    int iy_start = oy * str_h - pad_h;
    int ix_start = ox * str_w - pad_w;
    int iy_end = iy_start + kh;
    int ix_end = ix_start + kw;

    int iy0 = iy_start < 0 ? 0 : iy_start;
    int ix0 = ix_start < 0 ? 0 : ix_start;
    int iy1 = iy_end   > ih ? ih : iy_end;
    int ix1 = ix_end   > iw ? iw : ix_end;

    float acc = 0.0f;
    int count = 0;
    for (int ky = iy0; ky < iy1; ++ky) {
        for (int kx = ix0; kx < ix1; ++kx) {
            int in_idx = b_idx * c * ih * iw + c_idx * ih * iw + ky * iw + kx;
            acc += __input[in_idx];
            ++count;
        }
    }

    __output[gid] = (count > 0) ? (acc / (float)count) : 0.0f;
}

extern "C" __global__ void avg_pool2d_f16(
    const __half* __input,
    __half*       __output,
    int b, int c, int ih, int iw,
    int kh, int kw,
    int pad_h, int pad_w,
    int str_h, int str_w,
    int oh, int ow
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * c * oh * ow;
    if (gid >= total) return;

    int idx = gid;
    int b_idx = idx / (c * oh * ow); idx %= (c * oh * ow);
    int c_idx = idx / (oh * ow);     idx %= (oh * ow);
    int oy    = idx / ow;
    int ox    = idx % ow;

    int iy_start = oy * str_h - pad_h;
    int ix_start = ox * str_w - pad_w;
    int iy_end = iy_start + kh;
    int ix_end = ix_start + kw;

    int iy0 = iy_start < 0 ? 0 : iy_start;
    int ix0 = ix_start < 0 ? 0 : ix_start;
    int iy1 = iy_end   > ih ? ih : iy_end;
    int ix1 = ix_end   > iw ? iw : ix_end;

    float acc = 0.0f;
    int count = 0;
    for (int ky = iy0; ky < iy1; ++ky) {
        for (int kx = ix0; kx < ix1; ++kx) {
            int in_idx = b_idx * c * ih * iw + c_idx * ih * iw + ky * iw + kx;
            acc += __half2float(__input[in_idx]);
            ++count;
        }
    }

    __output[gid] = __float2half((count > 0) ? (acc / (float)count) : 0.0f);
}

// ─── max_pool2d ──────────────────────────────────────────────────────────────

extern "C" __global__ void max_pool2d_f32(
    const float* __input,
    float*       __output,
    int b, int c, int ih, int iw,
    int kh, int kw,
    int pad_h, int pad_w,
    int str_h, int str_w,
    int oh, int ow
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * c * oh * ow;
    if (gid >= total) return;

    int idx = gid;
    int b_idx = idx / (c * oh * ow); idx %= (c * oh * ow);
    int c_idx = idx / (oh * ow);     idx %= (oh * ow);
    int oy    = idx / ow;
    int ox    = idx % ow;

    int iy_start = oy * str_h - pad_h;
    int ix_start = ox * str_w - pad_w;
    int iy_end = iy_start + kh;
    int ix_end = ix_start + kw;

    int iy0 = iy_start < 0 ? 0 : iy_start;
    int ix0 = ix_start < 0 ? 0 : ix_start;
    int iy1 = iy_end   > ih ? ih : iy_end;
    int ix1 = ix_end   > iw ? iw : ix_end;

    float max_val = -INFINITY;
    for (int ky = iy0; ky < iy1; ++ky) {
        for (int kx = ix0; kx < ix1; ++kx) {
            int in_idx = b_idx * c * ih * iw + c_idx * ih * iw + ky * iw + kx;
            float v = __input[in_idx];
            if (v > max_val) max_val = v;
        }
    }

    __output[gid] = max_val;
}

extern "C" __global__ void max_pool2d_f16(
    const __half* __input,
    __half*       __output,
    int b, int c, int ih, int iw,
    int kh, int kw,
    int pad_h, int pad_w,
    int str_h, int str_w,
    int oh, int ow
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * c * oh * ow;
    if (gid >= total) return;

    int idx = gid;
    int b_idx = idx / (c * oh * ow); idx %= (c * oh * ow);
    int c_idx = idx / (oh * ow);     idx %= (oh * ow);
    int oy    = idx / ow;
    int ox    = idx % ow;

    int iy_start = oy * str_h - pad_h;
    int ix_start = ox * str_w - pad_w;
    int iy_end = iy_start + kh;
    int ix_end = ix_start + kw;

    int iy0 = iy_start < 0 ? 0 : iy_start;
    int ix0 = ix_start < 0 ? 0 : ix_start;
    int iy1 = iy_end   > ih ? ih : iy_end;
    int ix1 = ix_end   > iw ? iw : ix_end;

    float max_val = -INFINITY;
    for (int ky = iy0; ky < iy1; ++ky) {
        for (int kx = ix0; kx < ix1; ++kx) {
            int in_idx = b_idx * c * ih * iw + c_idx * ih * iw + ky * iw + kx;
            float v = __half2float(__input[in_idx]);
            if (v > max_val) max_val = v;
        }
    }

    __output[gid] = __float2half(max_val);
}