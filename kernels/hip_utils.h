#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

// Strided index: converts flat thread index to memory offset for non-contiguous tensors.
// dims_and_strides layout: [dim0, dim1, ..., stride0, stride1, ...]
__device__ __forceinline__ size_t get_strided_index(
    size_t idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t strided_i = 0;
    for (int d = (int)num_dims - 1; d >= 0; d--) {
        strided_i += (idx % dims[d]) * strides[d];
        idx /= dims[d];
    }
    return strided_i;
}

// Check if dims_and_strides is null (signals contiguous layout from Rust side)
__device__ __forceinline__ bool is_contiguous(const size_t *info) {
    return info == nullptr;
}

// Grid-stride loop bounds
#define GS_IDX (blockIdx.x * blockDim.x + threadIdx.x)
#define GS_STEP (blockDim.x * gridDim.x)

// Launch config helper: 256 threads/block, enough blocks to cover numel
#define LAUNCH_CFG(numel) dim3((((numel) + 255) / 256)), dim3(256)

// On Ubuntu-packaged ROCm, glibc <math.h> declares host-only math functions
// that conflict with HIP device intrinsics. Declare device-side versions
// using LLVM intrinsics and OCML (AMD's OpenCL Math Library).
#ifdef __HIP_DEVICE_COMPILE__
extern "C" {
// LLVM intrinsics (always available)
__device__ float expf(float) __asm__("llvm.exp.f32");
__device__ float logf(float) __asm__("llvm.log.f32");
__device__ float sinf(float) __asm__("llvm.sin.f32");
__device__ float cosf(float) __asm__("llvm.cos.f32");
__device__ float fabsf(float) __asm__("llvm.fabs.f32");
__device__ float ceilf(float) __asm__("llvm.ceil.f32");
__device__ float floorf(float) __asm__("llvm.floor.f32");
__device__ float roundf(float) __asm__("llvm.round.f32");
__device__ float truncf(float) __asm__("llvm.trunc.f32");
__device__ float fmaxf(float, float) __asm__("llvm.maxnum.f32");
__device__ float fminf(float, float) __asm__("llvm.minnum.f32");

// OCML functions (no LLVM intrinsic for these)
__device__ float __ocml_tanh_f32(float);
__device__ float __ocml_erf_f32(float);
__device__ float __ocml_pow_f32(float, float);
__device__ float __ocml_sqrt_f32(float);

// Inline wrappers: bypass glibc __host__ overloads.
// When the runtime JIT compiler uses -include hip/hip_runtime.h, __clang_hip_math.h
// already defines these. When compiling standalone (hipcc --genco), it doesn't.
// Use __clang_hip_math_h guard to detect which case we're in.
#ifndef __CLANG_HIP_MATH_H__
__device__ __forceinline__ float tanhf(float x) { return __ocml_tanh_f32(x); }
__device__ __forceinline__ float erff(float x) { return __ocml_erf_f32(x); }
__device__ __forceinline__ float powf(float x, float y) { return __ocml_pow_f32(x, y); }
__device__ __forceinline__ float sqrtf(float x) {
    float r;
    asm("v_sqrt_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}
#endif
}
#endif