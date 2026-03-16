//! Safe device management.

use crate::error::{check_hip, Result};
use hip_sys::hip_runtime;

#[derive(Debug, Clone)]
pub struct HipDevice {
    ordinal: usize,
}

impl HipDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        check_hip(unsafe { hip_runtime::hipSetDevice(ordinal as i32) })?;
        Ok(Self { ordinal })
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn set_current(&self) -> Result<()> {
        check_hip(unsafe { hip_runtime::hipSetDevice(self.ordinal as i32) })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.set_current()?;
        check_hip(unsafe { hip_runtime::hipDeviceSynchronize() })
    }

    pub fn device_count() -> Result<usize> {
        let mut count: i32 = 0;
        check_hip(unsafe { hip_runtime::hipGetDeviceCount(&mut count) })?;
        Ok(count as usize)
    }

    pub fn name(&self) -> Result<String> {
        let mut prop = std::mem::MaybeUninit::<hip_runtime::hipDeviceProp_t>::zeroed();
        check_hip(unsafe {
            hip_runtime::hipGetDeviceProperties(prop.as_mut_ptr(), self.ordinal as i32)
        })?;
        let prop = unsafe { prop.assume_init() };
        let name = unsafe { std::ffi::CStr::from_ptr(prop.name.as_ptr()) };
        Ok(name.to_string_lossy().into_owned())
    }

    pub fn total_memory(&self) -> Result<usize> {
        let mut prop = std::mem::MaybeUninit::<hip_runtime::hipDeviceProp_t>::zeroed();
        check_hip(unsafe {
            hip_runtime::hipGetDeviceProperties(prop.as_mut_ptr(), self.ordinal as i32)
        })?;
        let prop = unsafe { prop.assume_init() };
        Ok(prop.total_global_mem)
    }
}
