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
}

impl Drop for RocBlas {
    fn drop(&mut self) {
        unsafe { rocblas::rocblas_destroy_handle(self.handle) };
    }
}
