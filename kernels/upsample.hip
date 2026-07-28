#include "hip_utils.h"
#include <hip/hip_fp16.h>

// UpSample kernels: nearest1d, nearest2d, bilinear2d — F32 + F16
// nearest: output[oy][ox] = input[floor(oy * ih / oh)][floor(ox * iw / ow)]
// bilinear: bilinear interpolation on a 2×2 patch

// ─── upsample_nearest1d ────────────────────────────────────────────────────

extern "C" __global__ void upsample_nearest1d_f32(
    const float* __input,
    float*       __output,
    int n, int c, int i_len,
    int o_len
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * o_len;
    if (gid >= total) return;

    int idx = gid;
    int n_idx = idx / (c * o_len); idx %= (c * o_len);
    int c_idx = idx / o_len;
    int ox    = idx % o_len;

    int ix = (o_len == 1) ? 0 : (ox * i_len) / o_len;
    int in_idx = n_idx * c * i_len + c_idx * i_len + ix;
    __output[gid] = __input[in_idx];
}

extern "C" __global__ void upsample_nearest1d_f16(
    const __half* __input,
    __half*       __output,
    int n, int c, int i_len,
    int o_len
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * o_len;
    if (gid >= total) return;

    int idx = gid;
    int n_idx = idx / (c * o_len); idx %= (c * o_len);
    int c_idx = idx / o_len;
    int ox    = idx % o_len;

    int ix = (o_len == 1) ? 0 : (ox * i_len) / o_len;
    int in_idx = n_idx * c * i_len + c_idx * i_len + ix;
    __output[gid] = __input[in_idx];
}

// ─── upsample_nearest2d ──────────────────────────────────────────────────────

extern "C" __global__ void upsample_nearest2d_f32(
    const float* __input,
    float*       __output,
    int n, int c, int ih, int iw,
    int oh, int ow
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * oh * ow;
    if (gid >= total) return;

    int idx = gid;
    int n_idx = idx / (c * oh * ow); idx %= (c * oh * ow);
    int c_idx = idx / (oh * ow);     idx %= (oh * ow);
    int oy    = idx / ow;
    int ox    = idx % ow;

    int iy = (oh == 1) ? 0 : (oy * ih) / oh;
    int ix = (ow == 1) ? 0 : (ox * iw) / ow;
    int in_idx = n_idx * c * ih * iw + c_idx * ih * iw + iy * iw + ix;
    __output[gid] = __input[in_idx];
}

extern "C" __global__ void upsample_nearest2d_f16(
    const __half* __input,
    __half*       __output,
    int n, int c, int ih, int iw,
    int oh, int ow
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * oh * ow;
    if (gid >= total) return;

    int idx = gid;
    int n_idx = idx / (c * oh * ow); idx %= (c * oh * ow);
    int c_idx = idx / (oh * ow);     idx %= (oh * ow);
    int oy    = idx / ow;
    int ox    = idx % ow;

    int iy = (oh == 1) ? 0 : (oy * ih) / oh;
    int ix = (ow == 1) ? 0 : (ox * iw) / ow;
    int in_idx = n_idx * c * ih * iw + c_idx * ih * iw + iy * iw + ix;
    __output[gid] = __input[in_idx];
}

// ─── upsample_bilinear2d ─────────────────────────────────────────────────────
// align_corners=true  → ix = ox * (iw-1)/(ow-1)
// align_corners=false → ix = (ox + 0.5) * iw / ow - 0.5
// Then bilinear interpolate from 4 neighbours.

extern "C" __global__ void upsample_bilinear2d_f32(
    const float* __input,
    float*       __output,
    int n, int c, int ih, int iw,
    int oh, int ow,
    int align_corners  // 1 = true, 0 = false
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * oh * ow;
    if (gid >= total) return;

    int idx = gid;
    int n_idx = idx / (c * oh * ow); idx %= (c * oh * ow);
    int c_idx = idx / (oh * ow);     idx %= (oh * ow);
    int oy    = idx / ow;
    int ox    = idx % ow;

    float ix_f, iy_f;
    if (align_corners) {
        ix_f = (ow == 1) ? 0.0f : (float)ox * (float)(iw - 1) / (float)(ow - 1);
        iy_f = (oh == 1) ? 0.0f : (float)oy * (float)(ih - 1) / (float)(oh - 1);
    } else {
        ix_f = ((float)ox + 0.5f) * (float)iw / (float)ow - 0.5f;
        iy_f = ((float)oy + 0.5f) * (float)ih / (float)oh - 0.5f;
    }

    int ix0 = (int)floorf(ix_f);
    int iy0 = (int)floorf(iy_f);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    if (ix0 < 0) ix0 = 0;
    if (iy0 < 0) iy0 = 0;
    if (ix1 >= iw) ix1 = iw - 1;
    if (iy1 >= ih) iy1 = ih - 1;

    float lx = ix_f - (float)ix0;
    float ly = iy_f - (float)iy0;

    int base = n_idx * c * ih * iw + c_idx * ih * iw;

    float v00 = __input[base + iy0 * iw + ix0];
    float v01 = __input[base + iy0 * iw + ix1];
    float v10 = __input[base + iy1 * iw + ix0];
    float v11 = __input[base + iy1 * iw + ix1];

    float top = v00 + lx * (v01 - v00);
    float bot = v10 + lx * (v11 - v10);
    __output[gid] = top + ly * (bot - top);
}

extern "C" __global__ void upsample_bilinear2d_f16(
    const __half* __input,
    __half*       __output,
    int n, int c, int ih, int iw,
    int oh, int ow,
    int align_corners
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * oh * ow;
    if (gid >= total) return;

    int idx = gid;
    int n_idx = idx / (c * oh * ow); idx %= (c * oh * ow);
    int c_idx = idx / (oh * ow);     idx %= (oh * ow);
    int oy    = idx / ow;
    int ox    = idx % ow;

    float ix_f, iy_f;
    if (align_corners) {
        ix_f = (ow == 1) ? 0.0f : (float)ox * (float)(iw - 1) / (float)(ow - 1);
        iy_f = (oh == 1) ? 0.0f : (float)oy * (float)(ih - 1) / (float)(oh - 1);
    } else {
        ix_f = ((float)ox + 0.5f) * (float)iw / (float)ow - 0.5f;
        iy_f = ((float)oy + 0.5f) * (float)ih / (float)oh - 0.5f;
    }

    int ix0 = (int)floorf(ix_f);
    int iy0 = (int)floorf(iy_f);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    if (ix0 < 0) ix0 = 0;
    if (iy0 < 0) iy0 = 0;
    if (ix1 >= iw) ix1 = iw - 1;
    if (iy1 >= ih) iy1 = ih - 1;

    float lx = ix_f - (float)ix0;
    float ly = iy_f - (float)iy0;

    int base = n_idx * c * ih * iw + c_idx * ih * iw;

    float v00 = __half2float(__input[base + iy0 * iw + ix0]);
    float v01 = __half2float(__input[base + iy0 * iw + ix1]);
    float v10 = __half2float(__input[base + iy1 * iw + ix0]);
    float v11 = __half2float(__input[base + iy1 * iw + ix1]);

    float top = v00 + lx * (v01 - v00);
    float bot = v10 + lx * (v11 - v10);
    __output[gid] = __float2half(top + ly * (bot - top));
}