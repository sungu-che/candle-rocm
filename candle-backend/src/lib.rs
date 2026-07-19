//! ROCm/HIP backend for the candle ML framework.
//!
//! This crate provides AMD GPU support via HIP/ROCm. Add it as a dependency
//! and use `candle_rocm::device()` to get a GPU-backed candle Device:
//!
//! ```rust,ignore
//! let dev = candle_rocm::device(0)?;
//! let t = candle_rocm::Tensor::zeros((2, 3), candle_rocm::DType::F32, &dev)?;
//! ```

// Re-export everything from candle-core so users only need one dependency.
pub use candle_core::*;

/// Create a ROCm device for the given GPU ordinal.
pub fn device(ordinal: usize) -> candle_core::Result<Device> {
    Device::new_rocm(ordinal)
}

/// Number of ROCm-capable GPUs visible to the runtime.
pub fn device_count() -> Result<usize> {
    hip_runtime::device::HipDevice::device_count()
        .map_err(|e| candle_core::Error::Msg(format!("{e}")))
}

/// Human-readable name of the GPU (e.g. "AMD Radeon RX 5700 XT").
pub fn device_name(ordinal: usize) -> Result<String> {
    let d = hip_runtime::device::HipDevice::new(ordinal)
        .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
    d.name().map_err(|e| candle_core::Error::Msg(format!("{e}")))
}

/// Total VRAM in bytes for the given GPU.
pub fn total_vram(ordinal: usize) -> Result<usize> {
    let d = hip_runtime::device::HipDevice::new(ordinal)
        .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
    d.total_memory()
        .map_err(|e| candle_core::Error::Msg(format!("{e}")))
}

/// Returns `(free_bytes, total_bytes)` for the given GPU.
/// Uses `hipMemGetInfo` for live VRAM availability.
pub fn mem_info(ordinal: usize) -> Result<(usize, usize)> {
    let d = hip_runtime::device::HipDevice::new(ordinal)
        .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
    d.mem_info()
        .map_err(|e| candle_core::Error::Msg(format!("{e}")))
}

/// Returns true if at least one ROCm device is available.
pub fn is_available() -> bool {
    device_count().map(|c| c > 0).unwrap_or(false)
}
