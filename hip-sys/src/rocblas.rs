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

extern "C" {
    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;

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
}
