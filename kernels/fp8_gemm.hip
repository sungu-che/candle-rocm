// Fused FP8 dequant + F16 GEMM — warp-per-output-element.
//
// Each warp (32 threads) computes ONE output element.
// Thread t in warp reads X[m, k+t], W_fp8[n, k+t] elementwise
// along the K dimension in steps of 32 — perfectly coalesced
// since all lanes read consecutive addresses.
// Warp shuffle reduces the K partial sums to thread 0, which
// writes the result.
//
// C[m, n] = scale[n] * Σ_k X[m, k] * fp8tof32(W_fp8[n, k])

#include "hip_utils.h"
#include <hip/hip_fp16.h>

// ── F8E4M3 → F32 via F16 hardware ────────────────────────────────────────
// F8E4M3: 1 sign + 4 exponent (bias 7) + 3 mantissa
// F16:    1 sign + 5 exponent (bias 15) + 10 mantissa
// Map: F16 exponent = F8E4 exponent + 8 (because 15-7=8)
//     F16 mantissa = F8E4 mantissa << 7 (aligned to high 3 bits)
__device__ inline float fp8e4m3_to_f32(uint8_t v) {
    uint16_t s  = ((uint16_t)(v & 0x80u)) << 8u;      // sign → bit 15
    uint16_t e  = (uint16_t)((v >> 3u) & 0x0Fu);       // exponent (4-bit)
    uint16_t m  = (uint16_t)(v & 0x07u);                // mantissa (3-bit)

    __half h;
    uint16_t bits;
    if (e == 0) {
        // Subnormal: value = (-1)^s * m * 2^(-9)
        // Cannot represent correctly as F16 subnormal/normal because the
        // exponent gap (2^(-9) vs 2^(-14) subnormal or 2^(-14) normal) is
        // too large — naive bit-shifting underflows by factor 2^15 ≈ 32768x,
        // effectively zeroing all near-zero weights and destroying the output.
        // Compute directly in F32 instead.
        float sign = (s ? -1.0f : 1.0f);
        return sign * (float)(m) * 0.001953125f;  // 0.001953125 = exp2f(-9)
    } else {
        int16_t e16 = (int16_t)e + 8;  // bias offset: 15 - 7 = 8
        if (e16 >= 31) {
            // Overflow → clip to F16 max
            bits = s | 0x7BFFu;
        } else {
            bits = s | ((uint16_t)e16 << 10u) | (m << 7u);
        }
    }
    *reinterpret_cast<uint16_t*>(&h) = bits;
    return __half2float(h);
}

// ── Warp reduce ──────────────────────────────────────────────────────────
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

extern "C" __global__ void fp8_gemm_f16(
    const uint8_t* __restrict__ W_fp8,  // (N, K) row-major FP8 weights
    const __half*  __restrict__ X,       // (M, K) row-major F16 input
    const float*   __restrict__ scale,   // (N,) per-channel scale
    __half*        __restrict__ Y,       // (M, N) F16 output
    int M,
    int N,
    int K
) {
    int n = blockIdx.x;  // output column  (0..N-1)
    int m = blockIdx.y;  // input row      (0..M-1)
    if (n >= N || m >= M) return;

    int lane = threadIdx.x;   // 0..31
    float acc = 0.0f;

    const uint8_t* w_row = W_fp8 + (size_t)n * K;
    const __half*  x_row = X     + (size_t)m * K;
    float s = scale[n];

    // Each lane processes every 32nd element, starting at `lane`.
    // Within a warp: thread 0 reads k=0, thread 1 reads k=1, ...
    // These are consecutive bytes in W[n, 0:32] → perfect coalescing.
    for (int k = lane; k < K; k += 32) {
        uint8_t w_byte = w_row[k];
        float   x_val  = __half2float(x_row[k]);
        float   w_val  = fp8e4m3_to_f32(w_byte) * s;
        acc += x_val * w_val;
    }

    // Warp reduce → lane 0 writes
    float total = warp_reduce_sum(acc);
    if (lane == 0) {
        Y[(size_t)m * N + n] = __float2half(total);
    }
}