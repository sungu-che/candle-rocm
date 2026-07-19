//! Typed GPU memory buffers with RAII.

use crate::error::{check_hip, Result};
use hip_sys::hip_runtime::{self, hipMemcpyKind};
use std::marker::PhantomData;

/// Typed GPU memory buffer with RAII.
pub struct DeviceBuffer<T> {
    ptr: *mut std::ffi::c_void,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate uninitialized GPU memory for `len` elements.
    pub fn alloc(len: usize) -> Result<Self> {
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipMalloc(&mut ptr, bytes) })?;
        Ok(Self { ptr, len, _phantom: PhantomData })
    }

    /// Allocate zero-initialized GPU memory.
    pub fn alloc_zeros(len: usize) -> Result<Self> {
        let buf = Self::alloc(len)?;
        let bytes = len * std::mem::size_of::<T>();
        check_hip(unsafe { hip_runtime::hipMemset(buf.ptr, 0, bytes) })?;
        Ok(buf)
    }

    /// Copy from host slice to device.
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let buf = Self::alloc(data.len())?;
        let bytes = data.len() * std::mem::size_of::<T>();
        check_hip(unsafe {
            hip_runtime::hipMemcpy(
                buf.ptr,
                data.as_ptr() as *const _,
                bytes,
                hipMemcpyKind::hipMemcpyHostToDevice,
            )
        })?;
        Ok(buf)
    }

    /// Copy device buffer back to host.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        // SAFETY: `result` is zero-filled as a placeholder; hipMemcpy fully
        // overwrites every element before the caller reads any of them.
        // This pattern avoids the need to heap-allocate a typed buffer on
        // the Rust side while still being safe.
        let mut result = vec![unsafe { std::mem::zeroed() }; self.len];
        let bytes = self.len * std::mem::size_of::<T>();
        check_hip(unsafe {
            hip_runtime::hipMemcpy(
                result.as_mut_ptr() as *mut _,
                self.ptr,
                bytes,
                hipMemcpyKind::hipMemcpyDeviceToHost,
            )
        })?;
        Ok(result)
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Raw void pointer for kernel params and rocBLAS.
    /// When the buffer is used as input to rocBLAS (which takes *const),
    /// cast via `as_void_ptr() as *const c_void`.
    pub fn as_void_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// Specialised methods for byte buffers (`DeviceBuffer<u8>`).
impl DeviceBuffer<u8> {
    /// Allocate `bytes.len()` bytes of GPU memory and copy the host
    /// data directly — no intermediate `Vec` or type cast.
    ///
    /// This is the fast path for uploading mmap'd safetensors weight
    /// slices to the GPU: the `&[u8]` comes straight from the mmap
    /// region and lands in VRAM in a single `hipMemcpy`.
    pub fn from_host_bytes(bytes: &[u8]) -> Result<Self> {
        let buf = Self::alloc(bytes.len())?;
        if !bytes.is_empty() {
            check_hip(unsafe {
                hip_runtime::hipMemcpy(
                    buf.ptr,
                    bytes.as_ptr() as *const _,
                    bytes.len(),
                    hipMemcpyKind::hipMemcpyHostToDevice,
                )
            })?;
        }
        Ok(buf)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hip_runtime::hipFree(self.ptr) };
        }
    }
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Send + Sync> Sync for DeviceBuffer<T> {}
