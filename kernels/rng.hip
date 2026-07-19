// Philox 4x32-10 counter-based RNG.
// Produces high-quality uniform random f32 values.
// Deterministic: same (seed, offset) always produces same output.

#include "hip_utils.h"

// Philox round function constants
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

struct philox4_result {
    unsigned int v[4];
};

// Single Philox round: hi*lo multiply-and-swap
__device__ static philox4_result philox4_round(philox4_result ctr, unsigned int key0, unsigned int key1) {
    unsigned int lo0 = ctr.v[0] * PHILOX_M0;
    unsigned int hi0 = __umulhi(ctr.v[0], PHILOX_M0);
    unsigned int lo1 = ctr.v[2] * PHILOX_M1;
    unsigned int hi1 = __umulhi(ctr.v[2], PHILOX_M1);

    philox4_result out;
    out.v[0] = hi1 ^ ctr.v[1] ^ key0;
    out.v[1] = lo1;
    out.v[2] = hi0 ^ ctr.v[3] ^ key1;
    out.v[3] = lo0;
    return out;
}

// Full Philox 4x32-10: 10 rounds with key injection
__device__ static philox4_result philox4x32_10(unsigned int ctr0, unsigned int ctr1,
                                                unsigned int ctr2, unsigned int ctr3,
                                                unsigned int key0, unsigned int key1) {
    philox4_result ctr = {{ctr0, ctr1, ctr2, ctr3}};
    for (int round = 0; round < 10; round++) {
        ctr = philox4_round(ctr, key0, key1);
        key0 += PHILOX_W0;
        key1 += PHILOX_W1;
    }
    return ctr;
}

// Convert uint32 to uniform float in (0, 1]
__device__ static float uint_to_uniform(unsigned int x) {
    // Map [0, 2^32) to (0, 1] — excludes 0, includes 1
    return (float)(x >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

extern "C" __global__ void rng_uniform_f32(float* output, unsigned int n,
                                            unsigned long long seed, unsigned long long offset) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread generates 4 values
    unsigned int base = tid * 4;

    if (base >= n) return;

    unsigned int key0 = (unsigned int)(seed);
    unsigned int key1 = (unsigned int)(seed >> 32);
    unsigned int ctr0 = (unsigned int)(offset + tid);
    unsigned int ctr1 = (unsigned int)((offset + tid) >> 32);

    philox4_result r = philox4x32_10(ctr0, ctr1, 0, 0, key0, key1);

    for (int j = 0; j < 4 && base + j < n; j++) {
        output[base + j] = uint_to_uniform(r.v[j]);
    }
}

// Box-Muller transform: pairs of uniforms → pairs of normals
extern "C" __global__ void rng_normal_f32(float* output, unsigned int n,
                                           unsigned long long seed, unsigned long long offset,
                                           float mean, float stddev) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base = tid * 4;

    if (base >= n) return;

    unsigned int key0 = (unsigned int)(seed);
    unsigned int key1 = (unsigned int)(seed >> 32);
    unsigned int ctr0 = (unsigned int)(offset + tid);
    unsigned int ctr1 = (unsigned int)((offset + tid) >> 32);

    philox4_result r = philox4x32_10(ctr0, ctr1, 0, 0, key0, key1);

    float u1 = uint_to_uniform(r.v[0]);
    float u2 = uint_to_uniform(r.v[1]);
    float u3 = uint_to_uniform(r.v[2]);
    float u4 = uint_to_uniform(r.v[3]);

    // Box-Muller: two pairs of uniforms → four normals
    float r1 = sqrtf(-2.0f * logf(u1));
    float theta1 = 2.0f * 3.14159265358979323846f * u2;
    float r2 = sqrtf(-2.0f * logf(u3));
    float theta2 = 2.0f * 3.14159265358979323846f * u4;

    float n0 = r1 * cosf(theta1) * stddev + mean;
    float n1 = r1 * sinf(theta1) * stddev + mean;
    float n2 = r2 * cosf(theta2) * stddev + mean;
    float n3 = r2 * sinf(theta2) * stddev + mean;

    if (base < n) output[base] = n0;
    if (base + 1 < n) output[base + 1] = n1;
    if (base + 2 < n) output[base + 2] = n2;
    if (base + 3 < n) output[base + 3] = n3;
}
