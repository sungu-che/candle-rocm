//! HIP module and kernel loading.

use crate::error::{check_hip, HipError, Result};
use crate::stream::HipStream;
use hip_sys::hip_runtime;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

#[derive(Clone)]
pub struct HipModule {
    module: hip_runtime::hipModule_t,
    functions: HashMap<String, hip_runtime::hipFunction_t>,
}

impl HipModule {
    /// Load a compiled .hsaco GPU binary.
    pub fn load(path: &Path) -> Result<Self> {
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let mut module = std::ptr::null_mut();
        check_hip(unsafe { hip_runtime::hipModuleLoad(&mut module, c_path.as_ptr()) })?;
        Ok(Self {
            module,
            functions: HashMap::new(),
        })
    }

    /// Get a kernel function by name. Caches the lookup.
    pub fn get_function(&mut self, name: &str) -> Result<hip_runtime::hipFunction_t> {
        if let Some(&func) = self.functions.get(name) {
            return Ok(func);
        }
        let c_name = CString::new(name).unwrap();
        let mut func = std::ptr::null_mut();
        check_hip(unsafe {
            hip_runtime::hipModuleGetFunction(&mut func, self.module, c_name.as_ptr())
        })?;
        self.functions.insert(name.to_string(), func);
        Ok(func)
    }

    /// Launch a kernel on the default (null) stream.
    pub unsafe fn launch(
        func: hip_runtime::hipFunction_t,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        check_hip(hip_runtime::hipModuleLaunchKernel(
            func,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shared_mem,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    }

    /// Launch a kernel on a specific stream (async, pipelined).
    pub unsafe fn launch_on_stream(
        func: hip_runtime::hipFunction_t,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
        stream: &HipStream,
    ) -> Result<()> {
        check_hip(hip_runtime::hipModuleLaunchKernel(
            func,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shared_mem,
            stream.as_raw(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    }
}

unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl Drop for HipModule {
    fn drop(&mut self) {
        unsafe { hip_runtime::hipModuleUnload(self.module) };
    }
}

/// Compile a .hip source file to .hsaco using hipcc.
pub fn compile_kernel(src: &Path, out: &Path, arch: &str) -> Result<()> {
    // Discover hipcc in priority order:
    // 1. $ROCM_PATH/bin/hipcc  (respects ROCM_PATH env, used by upstream installs)
    // 2. /opt/rocm/bin/hipcc   (upstream AMD installer)
    // 3. /usr/bin/hipcc        (Ubuntu/Debian apt package)
    // 4. hipcc                 (PATH fallback for custom installations)
    let rocm_path = std::env::var("ROCM_PATH")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "/opt/rocm".to_string());
    let hipcc = [
        format!("{}/bin/hipcc", rocm_path),
        "/opt/rocm/bin/hipcc".to_string(),
        "/usr/bin/hipcc".to_string(),
        "hipcc".to_string(),
    ]
    .into_iter()
    .find(|p| std::path::Path::new(p).exists())
    .unwrap_or_else(|| "hipcc".to_string());
    // On Ubuntu-packaged ROCm, the system math.h declares host-only functions
    // that conflict with HIP device math intrinsics. Use --rocm-path and
    // include the HIP wrapper to get proper device-side declarations.
    let status = std::process::Command::new(hipcc)
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
            "-O3",
            "--cuda-device-only",
            "-include", "hip/hip_runtime.h",
            "-o",
            out.to_str().unwrap(),
            src.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| HipError::KernelCompileFailed { msg: e.to_string() })?;
    if !status.success() {
        return Err(HipError::KernelCompileFailed {
            msg: format!("hipcc exited with {status}"),
        });
    }
    Ok(())
}
