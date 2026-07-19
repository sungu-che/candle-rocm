// GPU dequantization kernels for GGUF quantized models.
// Upload raw quantized bytes to GPU (compact), dequant to f16 on GPU.
// Avoids: CPU dequant → f16 upload (3.5x more data transfer).
//
// Q4_0 block: d(f16) + qs[16] → 32 f16 values
//   value_i = (qi - 8) * d   where qi = (qs[i/2] >> (4*(i%2))) & 0x0F
//
// Q8_0 block: d(f16) + qs[32] → 32 f16 values
//   value_i = qi * d

#include "hip_utils.h"
#include <hip/hip_fp16.h>

// Q4_0 dequant → f16
// Input: raw Q4_0 bytes (18 bytes per block, 32 values per block)
// Output: f16 tensor (num_blocks * 32 elements)
extern "C" __global__ void dequant_q4_0_f16(
    const uint8_t* __restrict__ input,   // raw Q4_0 blocks
    __half* __restrict__ output,         // f16 output
    int num_blocks                       // total number of Q4_0 blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    // Each Q4_0 block: 2 bytes scale (f16) + 16 bytes quantized
    const uint8_t* block = input + block_idx * 18;
    __half d;
    // First 2 bytes = f16 scale
    *reinterpret_cast<uint16_t*>(&d) = *reinterpret_cast<const uint16_t*>(block);
    const uint8_t* qs = block + 2;

    __half* out = output + block_idx * 32;
    // Dequant 32 values from 16 bytes (each byte = 2 nibbles)
    for (int i = 0; i < 16; i++) {
        uint8_t byte = qs[i];
        uint8_t lo = byte & 0x0F;
        uint8_t hi = (byte >> 4) & 0x0F;
        // Q4_0: value = (q - 8) * d
        out[i * 2]     = __hmul(__float2half((float)lo - 8.0f), d);
        out[i * 2 + 1] = __hmul(__float2half((float)hi - 8.0f), d);
    }
}

// Q8_0 dequant → f16
// Input: raw Q8_0 bytes (34 bytes per block, 32 values per block)
// Output: f16 tensor (num_blocks * 32 elements)
extern "C" __global__ void dequant_q8_0_f16(
    const uint8_t* __restrict__ input,   // raw Q8_0 blocks
    __half* __restrict__ output,         // f16 output
    int num_blocks                       // total number of Q8_0 blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    // Each Q8_0 block: 2 bytes scale (f16) + 32 bytes quantized
    const uint8_t* block = input + block_idx * 34;
    __half d;
    *reinterpret_cast<uint16_t*>(&d) = *reinterpret_cast<const uint16_t*>(block);
    const uint8_t* qs = block + 2;

    __half* out = output + block_idx * 32;
    for (int i = 0; i < 32; i++) {
        out[i] = __hmul(__float2half((float)qs[i]), d);
    }
}

// Q4_0 dequant → f32
extern "C" __global__ void dequant_q4_0_f32(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const uint8_t* block = input + block_idx * 18;
    __half d_half;
    *reinterpret_cast<uint16_t*>(&d_half) = *reinterpret_cast<const uint16_t*>(block);
    float d = __half2float(d_half);
    const uint8_t* qs = block + 2;

    float* out = output + block_idx * 32;
    for (int i = 0; i < 16; i++) {
        uint8_t byte = qs[i];
        out[i * 2]     = ((float)(byte & 0x0F) - 8.0f) * d;
        out[i * 2 + 1] = ((float)((byte >> 4) & 0x0F) - 8.0f) * d;
    }
}

// Q8_0 dequant → f32
extern "C" __global__ void dequant_q8_0_f32(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const uint8_t* block = input + block_idx * 34;
    __half d_half;
    *reinterpret_cast<uint16_t*>(&d_half) = *reinterpret_cast<const uint16_t*>(block);
    float d = __half2float(d_half);
    const uint8_t* qs = block + 2;

    float* out = output + block_idx * 32;
    for (int i = 0; i < 32; i++) {
        out[i] = (float)qs[i] * d;
    }
}
