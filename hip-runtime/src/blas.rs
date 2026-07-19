//! Safe GEMM wrapper around rocBLAS.

use crate::error::{check_rocblas, Result};
use crate::memory::DeviceBuffer;
use hip_sys::rocblas;

pub struct RocBlas {
    handle: rocblas::rocblas_handle,
}

impl RocBlas {
    pub fn new() -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        check_rocblas(unsafe { rocblas::rocblas_create_handle(&mut handle) })?;
        Ok(Self { handle })
    }

    /// C = alpha * op(A) * op(B) + beta * C
    /// All matrices in column-major. Dimensions: A is m*k, B is k*n, C is m*n.
    pub fn sgemm(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        lda: usize,
        b: &DeviceBuffer<f32>,
        ldb: usize,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        ldc: usize,
    ) -> Result<()> {
        let op = |t| if t {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(unsafe {
            rocblas::rocblas_sgemm(
                self.handle,
                op(trans_a), op(trans_b),
                m as i32, n as i32, k as i32,
                &alpha,
                a.as_void_ptr(), lda as i32,
                b.as_void_ptr(), ldb as i32,
                &beta,
                c.as_void_ptr() as *mut _, ldc as i32,
            )
        })
    }

    /// Batched strided SGEMM.
    pub fn sgemm_strided_batched(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        lda: usize,
        stride_a: i64,
        b: &DeviceBuffer<f32>,
        ldb: usize,
        stride_b: i64,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        let op = |t| if t {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(unsafe {
            rocblas::rocblas_sgemm_strided_batched(
                self.handle,
                op(trans_a), op(trans_b),
                m as i32, n as i32, k as i32,
                &alpha,
                a.as_void_ptr(), lda as i32, stride_a,
                b.as_void_ptr(), ldb as i32, stride_b,
                &beta,
                c.as_void_ptr() as *mut _, ldc as i32, stride_c,
                batch_count as i32,
            )
        })
    }

    /// Raw-pointer SGEMM for type-erased storage (e.g., `DeviceBuffer<u8>`).
    ///
    /// # Safety
    /// Caller must ensure pointers reference valid f32 GPU memory with correct dimensions.
    pub unsafe fn sgemm_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const std::ffi::c_void,
        lda: usize,
        b: *const std::ffi::c_void,
        ldb: usize,
        beta: f32,
        c: *mut std::ffi::c_void,
        ldc: usize,
    ) -> Result<()> {
        let op = |t| if t {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(rocblas::rocblas_sgemm(
            self.handle,
            op(trans_a), op(trans_b),
            m as i32, n as i32, k as i32,
            &alpha,
            a, lda as i32,
            b, ldb as i32,
            &beta,
            c, ldc as i32,
        ))
    }

    /// Raw-pointer batched strided SGEMM.
    ///
    /// # Safety
    /// Same requirements as `sgemm_raw`, plus stride/batch constraints.
    pub unsafe fn sgemm_strided_batched_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: *const std::ffi::c_void,
        lda: usize,
        stride_a: i64,
        b: *const std::ffi::c_void,
        ldb: usize,
        stride_b: i64,
        beta: f32,
        c: *mut std::ffi::c_void,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        let op = |t| if t {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(rocblas::rocblas_sgemm_strided_batched(
            self.handle,
            op(trans_a), op(trans_b),
            m as i32, n as i32, k as i32,
            &alpha,
            a, lda as i32, stride_a,
            b, ldb as i32, stride_b,
            &beta,
            c, ldc as i32, stride_c,
            batch_count as i32,
        ))
    }

    // ── HGEMM (f16) ──────────────────────────────────────────────────────

    /// Half-precision GEMM: C = alpha * op(A) * op(B) + beta * C.
    /// All matrices in f16. alpha/beta are passed as raw f16 bytes (2 bytes).
    ///
    /// # Safety
    /// Caller must ensure pointers reference valid f16 GPU memory with correct dims.
    pub unsafe fn hgemm_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: u16,  // f16 bits
        a: *const std::ffi::c_void,
        lda: usize,
        b: *const std::ffi::c_void,
        ldb: usize,
        beta: u16,   // f16 bits
        c: *mut std::ffi::c_void,
        ldc: usize,
    ) -> Result<()> {
        let op = |t| if t {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(rocblas::rocblas_hgemm(
            self.handle,
            op(trans_a), op(trans_b),
            m as i32, n as i32, k as i32,
            &alpha as *const u16 as *const std::ffi::c_void,
            a, lda as i32,
            b, ldb as i32,
            &beta as *const u16 as *const std::ffi::c_void,
            c, ldc as i32,
        ))
    }

    /// Half-precision batched strided GEMM.
    ///
    /// # Safety
    /// Same requirements as `hgemm_raw`, plus stride/batch constraints.
    pub unsafe fn hgemm_strided_batched_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: u16,
        a: *const std::ffi::c_void,
        lda: usize,
        stride_a: i64,
        b: *const std::ffi::c_void,
        ldb: usize,
        stride_b: i64,
        beta: u16,
        c: *mut std::ffi::c_void,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        let op = |t| if t {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(rocblas::rocblas_hgemm_strided_batched(
            self.handle,
            op(trans_a), op(trans_b),
            m as i32, n as i32, k as i32,
            &alpha as *const u16 as *const std::ffi::c_void,
            a, lda as i32, stride_a,
            b, ldb as i32, stride_b,
            &beta as *const u16 as *const std::ffi::c_void,
            c, ldc as i32, stride_c,
            batch_count as i32,
        ))
    }

    // ── GEMM_EX (mixed-precision: BF16 I/O, F32 compute) ─────────────────

    /// Extended GEMM with explicit data types.
    /// For BF16: set a_type=b_type=c_type=BF16_R, compute_type=F32.
    /// alpha/beta must match compute_type (f32 for f32 compute).
    ///
    /// # Safety
    /// Caller must ensure pointers reference valid GPU memory of the declared types.
    pub unsafe fn gemm_ex_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_type: rocblas::rocblas_datatype,
        lda: usize,
        b: *const std::ffi::c_void,
        b_type: rocblas::rocblas_datatype,
        ldb: usize,
        beta: *const std::ffi::c_void,
        c: *mut std::ffi::c_void,
        c_type: rocblas::rocblas_datatype,
        ldc: usize,
        compute_type: rocblas::rocblas_compute_type,
    ) -> Result<()> {
        let op = |t| if t {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(rocblas::rocblas_gemm_ex(
            self.handle,
            op(trans_a), op(trans_b),
            m as i32, n as i32, k as i32,
            alpha,
            a, a_type, lda as i32,
            b, b_type, ldb as i32,
            beta,
            c, c_type, ldc as i32,
            compute_type,
        ))
    }

    /// Extended batched strided GEMM with explicit data types.
    ///
    /// # Safety
    /// Same requirements as `gemm_ex_raw`, plus stride/batch constraints.
    pub unsafe fn gemm_strided_batched_ex_raw(
        &self,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_type: rocblas::rocblas_datatype,
        lda: usize,
        stride_a: i64,
        b: *const std::ffi::c_void,
        b_type: rocblas::rocblas_datatype,
        ldb: usize,
        stride_b: i64,
        beta: *const std::ffi::c_void,
        c: *mut std::ffi::c_void,
        c_type: rocblas::rocblas_datatype,
        ldc: usize,
        stride_c: i64,
        batch_count: usize,
        compute_type: rocblas::rocblas_compute_type,
    ) -> Result<()> {
        let op = |t| if t {
            rocblas::rocblas_operation::rocblas_operation_transpose
        } else {
            rocblas::rocblas_operation::rocblas_operation_none
        };

        check_rocblas(rocblas::rocblas_gemm_strided_batched_ex(
            self.handle,
            op(trans_a), op(trans_b),
            m as i32, n as i32, k as i32,
            alpha,
            a, a_type, lda as i32, stride_a,
            b, b_type, ldb as i32, stride_b,
            beta,
            c, c_type, ldc as i32, stride_c,
            batch_count as i32,
            compute_type,
        ))
    }
}

unsafe impl Send for RocBlas {}
unsafe impl Sync for RocBlas {}

impl Drop for RocBlas {
    fn drop(&mut self) {
        unsafe { rocblas::rocblas_destroy_handle(self.handle) };
    }
}
