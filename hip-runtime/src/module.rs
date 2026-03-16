//! HIP module and kernel loading.

use crate::error::{check_hip, HipError, Result};
use hip_sys::hip_runtime;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

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

    /// Launch a kernel.
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
}

impl Drop for HipModule {
    fn drop(&mut self) {
        unsafe { hip_runtime::hipModuleUnload(self.module) };
    }
}

/// Compile a .hip source file to .hsaco using hipcc.
pub fn compile_kernel(src: &Path, out: &Path, arch: &str) -> Result<()> {
    let status = std::process::Command::new("/opt/rocm/bin/hipcc")
        .args([
            "--genco",
            &format!("--offload-arch={arch}"),
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
