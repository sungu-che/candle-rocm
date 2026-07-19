//! rocBLAS FFI bindings.

use std::ffi::c_void;
use std::os::raw::c_int;

pub type rocblas_handle = *mut c_void;
pub type rocblas_status = c_int;

pub const ROCBLAS_STATUS_SUCCESS: rocblas_status = 0;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum rocblas_operation {
    rocblas_operation_none = 111,
    rocblas_operation_transpose = 112,
    rocblas_operation_conjugate_transpose = 113,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum rocblas_datatype {
    rocblas_datatype_f16_r = 150,
    rocblas_datatype_f32_r = 151,
    rocblas_datatype_f64_r = 152,
    rocblas_datatype_bf16_r = 168,
    rocblas_datatype_i8_r = 160,
    rocblas_datatype_i32_r = 162,
    rocblas_datatype_f8_r = 166,
    rocblas_datatype_bf8_r = 167,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum rocblas_compute_type {
    rocblas_compute_type_f32 = 300,
    rocblas_compute_type_f64 = 301,
    rocblas_compute_type_f32_fast_xf32 = 302,
    rocblas_compute_type_f32_fast_f16 = 303,
    rocblas_compute_type_f32_fast_bf16 = 304,
    rocblas_compute_type_f32_fast_i8 = 305,
}

extern "C" {
    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;

    // ── SGEMM (f32) ──────────────────────────────────────────────────────

    pub fn rocblas_sgemm(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const f32,
        c: *mut c_void,
        ldc: c_int,
    ) -> rocblas_status;

    pub fn rocblas_sgemm_strided_batched(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const c_void,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        ldb: c_int,
        stride_b: i64,
        beta: *const f32,
        c: *mut c_void,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
    ) -> rocblas_status;

    // ── HGEMM (f16) ──────────────────────────────────────────────────────

    /// Half-precision GEMM: C = alpha * op(A) * op(B) + beta * C.
    /// All of alpha, beta, A, B, C are in f16 (half).
    pub fn rocblas_hgemm(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void, // __half* (f16)
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const c_void, // __half* (f16)
        c: *mut c_void,
        ldc: c_int,
    ) -> rocblas_status;

    pub fn rocblas_hgemm_strided_batched(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void, // __half*
        a: *const c_void,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        ldb: c_int,
        stride_b: i64,
        beta: *const c_void, // __half*
        c: *mut c_void,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
    ) -> rocblas_status;

    // ── GEMM_EX (mixed-precision: BF16 I/O, F32 compute) ─────────────────

    /// Extended GEMM with explicit input/output/compute types.
    /// For BF16: a_type=b_type=c_type=bf16_r, compute_type=f32.
    pub fn rocblas_gemm_ex(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: rocblas_datatype,
        lda: c_int,
        b: *const c_void,
        b_type: rocblas_datatype,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        c_type: rocblas_datatype,
        ldc: c_int,
        compute_type: rocblas_compute_type,
    ) -> rocblas_status;

    pub fn rocblas_gemm_strided_batched_ex(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: rocblas_datatype,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        b_type: rocblas_datatype,
        ldb: c_int,
        stride_b: i64,
        beta: *const c_void,
        c: *mut c_void,
        c_type: rocblas_datatype,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
        compute_type: rocblas_compute_type,
    ) -> rocblas_status;
}
