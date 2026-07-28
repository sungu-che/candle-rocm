//! Typed GPU memory buffers with RAII.

use crate::error::{check_hip, Result};
use crate::stream::HipStream;
use hip_sys::hip_runtime::{self, hipMemcpyKind};
use std::marker::PhantomData;
use std::mem::MaybeUninit;

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
        let mut result: Vec<MaybeUninit<T>> = Vec::with_capacity(self.len);
        // Allocate uninitialized backing store
        let ptr = result.as_mut_ptr();
        std::mem::forget(result);

        let bytes = self.len * std::mem::size_of::<T>();
        check_hip(unsafe {
            hip_runtime::hipMemcpy(
                ptr as *mut _,
                self.ptr,
                bytes,
                hipMemcpyKind::hipMemcpyDeviceToHost,
            )
        })?;

        // SAFETY: hipMemcpy fully overwrote every element. Re-interpret from
        // MaybeUninit to initialized T. This matches the std library pattern
        // for reading from potentially-uninitialized memory that was actually
        // fully initialized by the copy.
        let result = unsafe { Vec::from_raw_parts(ptr as *mut T, self.len, self.len) };
        Ok(result)
    }

    /// Asynchronously copy from host slice to device.
    /// The copy is enqueued on the given stream and may not be complete
    /// when this function returns. Use `stream.synchronize()` to wait.
    pub fn from_slice_async(data: &[T], stream: &HipStream) -> Result<Self> {
        let buf = Self::alloc(data.len())?;
        let bytes = data.len() * std::mem::size_of::<T>();
        check_hip(unsafe {
            hip_runtime::hipMemcpyAsync(
                buf.ptr,
                data.as_ptr() as *const _,
                bytes,
                hipMemcpyKind::hipMemcpyHostToDevice,
                stream.as_raw(),
            )
        })?;
        Ok(buf)
    }

    /// Asynchronously copy device buffer back to host.
    /// The copy is enqueued on the given stream and may not be complete
    /// when this function returns. Use `stream.synchronize()` to wait.
    pub fn to_vec_async(&self, stream: &HipStream) -> Result<Vec<T>> {
        let mut result: Vec<MaybeUninit<T>> = Vec::with_capacity(self.len);
        let ptr = result.as_mut_ptr();
        std::mem::forget(result);

        let bytes = self.len * std::mem::size_of::<T>();
        check_hip(unsafe {
            hip_runtime::hipMemcpyAsync(
                ptr as *mut _,
                self.ptr,
                bytes,
                hipMemcpyKind::hipMemcpyDeviceToHost,
                stream.as_raw(),
            )
        })?;

        let result = unsafe { Vec::from_raw_parts(ptr as *mut T, self.len, self.len) };
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

    /// Async version of `from_host_bytes`. Enqueues the copy on a stream.
    pub fn from_host_bytes_async(bytes: &[u8], stream: &HipStream) -> Result<Self> {
        let buf = Self::alloc(bytes.len())?;
        if !bytes.is_empty() {
            check_hip(unsafe {
                hip_runtime::hipMemcpyAsync(
                    buf.ptr,
                    bytes.as_ptr() as *const _,
                    bytes.len(),
                    hipMemcpyKind::hipMemcpyHostToDevice,
                    stream.as_raw(),
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
