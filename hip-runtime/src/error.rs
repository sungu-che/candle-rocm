//! Error types for HIP runtime operations.

use std::fmt;

#[derive(Debug)]
pub enum HipError {
    HipRuntimeError { code: i32, msg: String },
    RocblasError { code: i32 },
    KernelNotFound { name: String },
    KernelCompileFailed { msg: String },
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HipRuntimeError { code, msg } => write!(f, "HIP error {code}: {msg}"),
            Self::RocblasError { code } => write!(f, "rocBLAS error {code}"),
            Self::KernelNotFound { name } => write!(f, "kernel not found: {name}"),
            Self::KernelCompileFailed { msg } => write!(f, "kernel compile failed: {msg}"),
        }
    }
}

impl std::error::Error for HipError {}

pub type Result<T> = std::result::Result<T, HipError>;

/// Check a HIP status code and convert to Result.
pub fn check_hip(code: i32) -> Result<()> {
    if code == hip_sys::hip_runtime::HIP_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = hip_sys::hip_runtime::hipGetErrorString(code);
            if ptr.is_null() {
                "unknown error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(HipError::HipRuntimeError { code, msg })
    }
}

pub fn check_rocblas(code: i32) -> Result<()> {
    if code == hip_sys::rocblas::ROCBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(HipError::RocblasError { code })
    }
}
