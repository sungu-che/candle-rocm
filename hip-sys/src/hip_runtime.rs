//! HIP runtime FFI bindings.

use std::ffi::c_void;
use std::os::raw::c_int;

pub type hipError_t = c_int;
pub type hipDevice_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipEvent_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipDeviceptr_t = *mut c_void;

pub const HIP_SUCCESS: hipError_t = 0;
pub const hipErrorNotReady: hipError_t = 2;

// Stream flags for hipStreamCreateWithFlags
pub const hipStreamDefault: u32 = 0;
pub const hipStreamNonBlocking: u32 = 1;

// Event flags for hipEventCreateWithFlags
pub const hipEventDefault: u32 = 0;
pub const hipEventBlockingSync: u32 = 1;
pub const hipEventDisableTiming: u32 = 2;
pub const hipEventInterprocess: u32 = 4;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
}

#[repr(C)]
#[derive(Debug)]
pub struct hipDeviceProp_t {
    pub name: [std::os::raw::c_char; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    // hipDeviceProp_t has 100+ fields in ROCm 6.x (~792+ bytes total).
    // We over-allocate to avoid writing past the struct boundary.
    pub _padding: [u8; 4096],
}

extern "C" {
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipGetDevice(device_id: *mut c_int) -> hipError_t;
    pub fn hipGetDeviceProperties(prop: *mut hipDeviceProp_t, device_id: c_int) -> hipError_t;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;
    pub fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipModuleLoad(
        module: *mut hipModule_t,
        fname: *const std::os::raw::c_char,
    ) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        func: *mut hipFunction_t,
        module: hipModule_t,
        name: *const std::os::raw::c_char,
    ) -> hipError_t;
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;

    /// Query free and total device memory in bytes.
    /// Both pointers must be valid. Returns HIP_SUCCESS on success.
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const std::os::raw::c_char;

    // ── Streams ─────────────────────────────────────────────────────────
    /// Create an asynchronous stream.
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    /// Create a stream with the specified flags.
    pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: u32) -> hipError_t;
    /// Destroy a stream.
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    /// Wait for a stream to complete.
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    /// Query if a stream is complete.
    pub fn hipStreamQuery(stream: hipStream_t) -> hipError_t;

    // ── Events ──────────────────────────────────────────────────────────
    /// Create a GPU event.
    pub fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
    /// Create a GPU event with the specified flags.
    pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: u32) -> hipError_t;
    /// Destroy a GPU event.
    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
    /// Record an event on the given stream.
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    /// Wait for an event to complete.
    pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
    /// Query if an event has been recorded.
    pub fn hipEventQuery(event: hipEvent_t) -> hipError_t;
    /// Get elapsed time between two events in milliseconds.
    pub fn hipEventElapsedTime(ms: *mut f32, start: hipEvent_t, end: hipEvent_t) -> hipError_t;

    // ── Async memory ops ────────────────────────────────────────────────
    /// Asynchronous H2D copy.
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;
    /// Asynchronous device memory set.
    pub fn hipMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        size: usize,
        stream: hipStream_t,
    ) -> hipError_t;
}
