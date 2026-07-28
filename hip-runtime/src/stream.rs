//! Asynchronous streams and events for pipelined GPU operations.

use crate::error::{check_hip, Result};
use hip_sys::hip_runtime;

/// An asynchronous GPU stream for pipelined operations.
///
/// A stream is a sequence of operations that execute in order on the GPU.
/// Unlike the default stream, explicit streams allow launching multiple
/// independent workloads that can overlap (e.g., compute + memory transfer).
///
/// Streams are created per-device and automatically set the active device.
#[derive(Debug, Clone)]
pub struct HipStream {
    stream: hip_runtime::hipStream_t,
}

unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    /// Create a new stream with default flags.
    pub fn new() -> Result<Self> {
        let mut stream = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipStreamCreate(&mut stream) })?;
        Ok(Self { stream })
    }

    /// Create a new stream with explicit flags.
    ///
    /// - `hip_stream_default` (0): default behavior, may synchronize with other streams
    /// - `hip_stream_non_blocking` (1): does not synchronize with other streams
    pub fn with_flags(flags: u32) -> Result<Self> {
        let mut stream = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipStreamCreateWithFlags(&mut stream, flags) })?;
        Ok(Self { stream })
    }

    /// Create a non-blocking stream (doesn't synchronize with default stream).
    pub fn non_blocking() -> Result<Self> {
        Self::with_flags(hip_runtime::hipStreamNonBlocking)
    }

    /// Returns the raw HIP stream handle.
    pub fn as_raw(&self) -> hip_runtime::hipStream_t {
        self.stream
    }

    /// Block until all operations in this stream complete.
    pub fn synchronize(&self) -> Result<()> {
        check_hip(unsafe { hip_runtime::hipStreamSynchronize(self.stream) })
    }

    /// Query if the stream has completed all queued operations.
    /// Returns `Ok(true)` if done, `Ok(false)` if still running.
    /// Never blocks — use `synchronize()` if you need to block.
    pub fn is_done(&self) -> Result<bool> {
        let code = unsafe { hip_runtime::hipStreamQuery(self.stream) };
        if code == hip_runtime::HIP_SUCCESS {
            Ok(true)
        } else if code == hip_runtime::hipErrorNotReady as i32 {
            Ok(false)
        } else {
            check_hip(code).map(|_| false)
        }
    }

    /// Default (null) stream handle for passing to APIs that accept an optional stream.
    pub fn null() -> hip_runtime::hipStream_t {
        std::ptr::null_mut()
    }

    /// Returns true if this is the null/default stream.
    pub fn is_null(&self) -> bool {
        self.stream.is_null()
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            // Ignore errors in Drop — HIP may already be shut down.
            let _ = check_hip(unsafe { hip_runtime::hipStreamDestroy(self.stream) });
        }
    }
}

impl Default for HipStream {
    fn default() -> Self {
        // Return a handle to the null stream (no cleanup needed).
        Self { stream: std::ptr::null_mut() }
    }
}

/// A GPU event for measuring elapsed time between two points in the stream,
/// or for signaling/completion notification across streams.
///
/// Events are recorded on a stream and can be used to:
/// - Measure elapsed time between two events (GPU-only, no CPU overhead)
/// - Make another stream wait until the event is recorded (stream synchronization)
/// - Poll for completion without blocking
#[derive(Debug)]
pub struct HipEvent {
    event: hip_runtime::hipEvent_t,
    owns: bool,
}

unsafe impl Send for HipEvent {}
unsafe impl Sync for HipEvent {}

impl HipEvent {
    /// Create a new event with default flags (timing-enabled, blocking sync).
    pub fn new() -> Result<Self> {
        let mut event = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipEventCreate(&mut event) })?;
        Ok(Self { event, owns: true })
    }

    /// Create an event with explicit flags.
    ///
    /// - `hip_event_default` (0): enables timing, blocking sync
    /// - `hip_event_disable_timing` (2): faster, no timing support (good for signaling)
    /// - `hip_event_blocking_sync` (1): `event.synchronize()` blocks instead of spinning
    pub fn with_flags(flags: u32) -> Result<Self> {
        let mut event = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipEventCreateWithFlags(&mut event, flags) })?;
        Ok(Self { event, owns: true })
    }

    /// Create a timing-disabled event (faster for signaling).
    pub fn signaling() -> Result<Self> {
        Self::with_flags(hip_runtime::hipEventDisableTiming)
    }

    /// Record this event on the given stream.
    /// The event records the current position of the stream.
    pub fn record(&self, stream: &HipStream) -> Result<()> {
        check_hip(unsafe { hip_runtime::hipEventRecord(self.event, stream.as_raw()) })
    }

    /// Block until this event is recorded and all subsequent stream ops complete.
    pub fn synchronize(&self) -> Result<()> {
        check_hip(unsafe { hip_runtime::hipEventSynchronize(self.event) })
    }

    /// Query if the event has been recorded.
    /// Returns `Ok(true)` if recorded, `Ok(false)` if still pending.
    /// Never blocks — use `synchronize()` if you need to block.
    pub fn is_complete(&self) -> Result<bool> {
        let code = unsafe { hip_runtime::hipEventQuery(self.event) };
        if code == hip_runtime::HIP_SUCCESS {
            Ok(true)
        } else if code == hip_runtime::hipErrorNotReady as i32 {
            Ok(false)
        } else {
            check_hip(code).map(|_| false)
        }
    }

    /// Get elapsed time in milliseconds between two recorded events.
    ///
    /// Both events must have been recorded on the same device.
    /// Both events must have timing enabled (i.e., not created with
    /// `hip_event_disable_timing`).
    pub fn elapsed_ms(&self, start: &HipEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        check_hip(unsafe {
            hip_runtime::hipEventElapsedTime(&mut ms, start.event, self.event)
        })?;
        Ok(ms)
    }

    /// Returns the raw HIP event handle.
    pub fn as_raw(&self) -> hip_runtime::hipEvent_t {
        self.event
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if self.owns && !self.event.is_null() {
            let _ = check_hip(unsafe { hip_runtime::hipEventDestroy(self.event) });
        }
    }
}

impl Default for HipEvent {
    fn default() -> Self {
        Self {
            event: std::ptr::null_mut(),
            owns: false,
        }
    }
}