use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use float8::F8E4M3;
use hip_runtime::blas::RocBlas;
use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;
use hip_runtime::module::{compile_kernel, HipModule};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

mod storage;
#[cfg(feature = "rocm")]
pub mod gpu_fp8gemm;

/// A ROCm GPU device.
#[derive(Clone, Debug)]
pub struct RocmDevice {
    ordinal: usize,
    inner: Arc<Mutex<RocmDeviceInner>>,
}

struct RocmDeviceInner {
    _device: HipDevice,
    blas: RocBlas,
    modules: HashMap<String, HipModule>,
    kernel_dir: PathBuf,
    arch: String,
    rng_seed: u64,
    rng_offset: u64,
}

impl std::fmt::Debug for RocmDeviceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmDeviceInner")
            .field("kernel_dir", &self.kernel_dir)
            .field("arch", &self.arch)
            .finish_non_exhaustive()
    }
}

/// GPU-resident tensor storage.
pub struct RocmStorage {
    pub(crate) buf: DeviceBuffer<u8>,
    pub(crate) dtype: DType,
    pub(crate) device: RocmDevice,
}

// Safety: ROCm device memory is managed by the HIP runtime.
// DeviceBuffer holds a raw pointer but owns the allocation.
unsafe impl Send for RocmStorage {}
unsafe impl Sync for RocmStorage {}

impl std::fmt::Debug for RocmStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmStorage")
            .field("dtype", &self.dtype)
            .field("byte_size", &self.buf.byte_size())
            .field("device", &self.device)
            .finish()
    }
}

impl RocmStorage {
    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buf.as_mut_ptr()
    }

    pub fn elem_count(&self) -> usize {
        self.buf.byte_size() / self.dtype.size_in_bytes()
    }

    /// Replace GPU buffer contents from a CpuStorage (same dtype required).
    /// Used by inplace custom ops CPU fallback.
    pub fn replace_from_cpu_storage(&mut self, cpu: &CpuStorage) -> Result<()> {
        let (bytes, _dtype) = cpu_storage_to_bytes(cpu.clone());
        let new_buf = DeviceBuffer::from_slice(&bytes)
            .map_err(|e| crate::Error::Msg(format!("upload failed: {e}")))?;
        self.buf = new_buf;
        Ok(())
    }
}

fn dtype_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::U8 => "u8",
        DType::U32 => "u32",
        DType::I64 => "i64",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F8E4M3 => "f8e4m3",
        DType::F6E2M3 => "f6e2m3",
        DType::F6E3M2 => "f6e3m2",
        DType::F4 => "f4",
        DType::F8E8M0 => "f8e8m0",
    }
}

impl RocmDevice {
    fn get_or_compile(&self, kernel_name: &str) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if inner.modules.contains_key(kernel_name) {
            return Ok(());
        }
        let src = inner.kernel_dir.join(format!("{kernel_name}.hip"));
        let out = inner.kernel_dir.join(format!("{kernel_name}.hsaco"));
        if !out.exists() {
            compile_kernel(&src, &out, &inner.arch)
                .map_err(|e| crate::Error::Msg(format!("kernel compile failed: {e}")))?;
        }
        let module = HipModule::load(&out)
            .map_err(|e| crate::Error::Msg(format!("module load failed: {e}")))?;
        inner.modules.insert(kernel_name.to_string(), module);
        Ok(())
    }

    pub(crate) fn with_module<F, R>(&self, kernel_file: &str, f: F) -> Result<R>
    where
        F: FnOnce(&mut HipModule, &RocBlas) -> Result<R>,
    {
        self.get_or_compile(kernel_file)?;
        let mut inner = self.inner.lock().unwrap();
        // Borrow disjoint fields to satisfy the borrow checker:
        // modules is borrowed mutably, blas is borrowed immutably.
        let inner_ref = &mut *inner;
        let module = inner_ref.modules.get_mut(kernel_file).unwrap();
        f(module, &inner_ref.blas)
    }

    fn with_blas<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&RocBlas) -> Result<R>,
    {
        let inner = self.inner.lock().unwrap();
        f(&inner.blas)
    }

    /// Returns `(free_bytes, total_bytes)` for this device.
    /// Uses `hipMemGetInfo` for live VRAM availability.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let inner = self.inner.lock().unwrap();
        inner
            ._device
            .mem_info()
            .map_err(|e| crate::Error::Msg(format!("mem_info failed: {e}")))
    }

    /// Returns the GPU device name (e.g. "AMD Radeon RX 6900 XT").
    pub fn gpu_name(&self) -> Result<String> {
        let inner = self.inner.lock().unwrap();
        inner
            ._device
            .name()
            .map_err(|e| crate::Error::Msg(format!("device name failed: {e}")))
    }

    fn alloc_buf(&self, byte_size: usize) -> Result<DeviceBuffer<u8>> {
        DeviceBuffer::<u8>::alloc(byte_size)
            .map_err(|e| crate::Error::Msg(format!("GPU alloc failed: {e}")))
    }

    fn alloc_zeros_buf(&self, byte_size: usize) -> Result<DeviceBuffer<u8>> {
        DeviceBuffer::<u8>::alloc_zeros(byte_size)
            .map_err(|e| crate::Error::Msg(format!("GPU alloc_zeros failed: {e}")))
    }

    /// Direct upload of a raw byte slice (e.g. from an mmap'd safetensors
    /// tensor view) to a GPU buffer, skipping the `CpuStorage` and
    /// `cpu_storage_to_bytes` intermediates.
    ///
    /// `bytes` must already be in the correct on-device layout (little-endian,
    /// matching `dtype`). The caller is responsible for ensuring alignment;
    /// safetensors tensor data is naturally aligned.
    pub(crate) fn storage_from_bytes(&self, bytes: &[u8], dtype: DType) -> Result<RocmStorage> {
        let buf = DeviceBuffer::<u8>::from_host_bytes(bytes)
            .map_err(|e| crate::Error::Msg(format!("GPU from_host_bytes failed: {e}")))?;
        Ok(RocmStorage {
            buf,
            dtype,
            device: self.clone(),
        })
    }

    fn upload_info(&self, layout: &Layout) -> Result<Option<DeviceBuffer<usize>>> {
        if layout.is_contiguous() {
            return Ok(None);
        }
        let info: Vec<usize> = [layout.dims(), layout.stride()].concat();
        let buf = DeviceBuffer::from_slice(&info)
            .map_err(|e| crate::Error::Msg(format!("info upload failed: {e}")))?;
        Ok(Some(buf))
    }

    fn upload_binary_info(
        &self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Option<DeviceBuffer<usize>>> {
        if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            return Ok(None);
        }
        let info: Vec<usize> = [lhs_l.dims(), lhs_l.stride(), rhs_l.stride()].concat();
        let buf = DeviceBuffer::from_slice(&info)
            .map_err(|e| crate::Error::Msg(format!("info upload failed: {e}")))?;
        Ok(Some(buf))
    }
}

fn launch_cfg(numel: usize) -> (u32, u32) {
    let block = 256u32;
    let grid = ((numel as u32 + block - 1) / block).max(1);
    (grid, block)
}

impl crate::backend::BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = HipDevice::new(ordinal)
            .map_err(|e| crate::Error::Msg(format!("HipDevice::new failed: {e}")))?;
        let blas = RocBlas::new()
            .map_err(|e| crate::Error::Msg(format!("RocBlas::new failed: {e}")))?;

        let kernel_dir = std::env::var("CANDLE_ROCM_KERNELS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../kernels")
            });
        let arch = std::env::var("HIP_ARCH").unwrap_or_else(|_| "gfx1030".to_string());

        Ok(Self {
            ordinal,
            inner: Arc::new(Mutex::new(RocmDeviceInner {
                _device: device,
                blas,
                modules: HashMap::new(),
                kernel_dir,
                arch,
                rng_seed: 299792458,
                rng_offset: 0,
            })),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.ordinal == rhs.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let byte_size = shape.elem_count() * dtype.size_in_bytes();
        let buf = self.alloc_zeros_buf(byte_size)?;
        Ok(RocmStorage {
            buf,
            dtype,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let byte_size = shape.elem_count() * dtype.size_in_bytes();
        let buf = self.alloc_buf(byte_size)?;
        Ok(RocmStorage {
            buf,
            dtype,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let cpu_storage = T::to_cpu_storage(data);
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(storage.clone())
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        let (bytes, dtype) = cpu_storage_to_bytes(storage);
        let buf = DeviceBuffer::from_slice(&bytes)
            .map_err(|e| crate::Error::Msg(format!("upload failed: {e}")))?;
        Ok(RocmStorage {
            buf,
            dtype,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        // Generate on CPU and upload for now
        let cpu_storage = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, lo, hi)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        let cpu_storage = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.rng_seed = seed;
        inner.rng_offset = 0;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        let inner = self.inner.lock().unwrap();
        Ok(inner.rng_seed)
    }

    fn synchronize(&self) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        inner
            ._device
            .synchronize()
            .map_err(|e| crate::Error::Msg(format!("synchronize failed: {e}")))
    }
}

fn cpu_storage_to_bytes(storage: CpuStorage) -> (Vec<u8>, DType) {
    match storage {
        CpuStorage::U8(v) => (v, DType::U8),
        CpuStorage::U32(v) => {
            let bytes: Vec<u8> = v
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            (bytes, DType::U32)
        }
        CpuStorage::I64(v) => {
            let bytes: Vec<u8> = v
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            (bytes, DType::I64)
        }
        CpuStorage::BF16(v) => {
            let bytes: Vec<u8> = v
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            (bytes, DType::BF16)
        }
        CpuStorage::F16(v) => {
            let bytes: Vec<u8> = v
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            (bytes, DType::F16)
        }
        CpuStorage::F32(v) => {
            let bytes: Vec<u8> = v
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            (bytes, DType::F32)
        }
        CpuStorage::F64(v) => {
            let bytes: Vec<u8> = v
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            (bytes, DType::F64)
        }
        CpuStorage::I16(v) => {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            (bytes, DType::I16)
        }
        CpuStorage::I32(v) => {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            (bytes, DType::I32)
        }
        CpuStorage::F8E4M3(v) => {
            let bytes: Vec<u8> = v.iter().map(|x| { let b: u8 = unsafe { std::mem::transmute(*x) }; b }).collect();
            (bytes, DType::F8E4M3)
        }
        CpuStorage::F6E2M3(v) => {
            let bytes: Vec<u8> = v.iter().map(|x| { let b: u8 = unsafe { std::mem::transmute(*x) }; b }).collect();
            (bytes, DType::F6E2M3)
        }
        CpuStorage::F6E3M2(v) => {
            let bytes: Vec<u8> = v.iter().map(|x| { let b: u8 = unsafe { std::mem::transmute(*x) }; b }).collect();
            (bytes, DType::F6E3M2)
        }
        CpuStorage::F4(v) => {
            let bytes: Vec<u8> = v.iter().map(|x| { let b: u8 = unsafe { std::mem::transmute(*x) }; b }).collect();
            (bytes, DType::F4)
        }
        CpuStorage::F8E8M0(v) => {
            let bytes: Vec<u8> = v.iter().map(|x| { let b: u8 = unsafe { std::mem::transmute(*x) }; b }).collect();
            (bytes, DType::F8E8M0)
        }
    }
}

fn bytes_to_cpu_storage(bytes: &[u8], dtype: DType) -> CpuStorage {
    match dtype {
        DType::U8 => CpuStorage::U8(bytes.to_vec()),
        DType::U32 => {
            let v: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            CpuStorage::U32(v)
        }
        DType::I64 => {
            let v: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            CpuStorage::I64(v)
        }
        DType::BF16 => {
            let v: Vec<half::bf16> = bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()))
                .collect();
            CpuStorage::BF16(v)
        }
        DType::F16 => {
            let v: Vec<half::f16> = bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()))
                .collect();
            CpuStorage::F16(v)
        }
        DType::F32 => {
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            CpuStorage::F32(v)
        }
        DType::F64 => {
            let v: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            CpuStorage::F64(v)
        }
        DType::F8E4M3 => {
            // F8E4M3 is a 1-byte float; CpuStorage stores it as Vec<F8E4M3>.
            // The `float8::F8E4M3` type is `#[repr(transparent)]` over `u8`,
            // so a byte slice is bit-identical to a `&[F8E4M3]`.
            let v: Vec<float8::F8E4M3> = bytes
                .iter()
                .map(|b| unsafe { std::mem::transmute::<u8, float8::F8E4M3>(*b) })
                .collect();
            CpuStorage::F8E4M3(v)
        }
        DType::F8E8M0 => CpuStorage::F8E8M0(bytes.to_vec()),
        _ => panic!("unsupported dtype for bytes_to_cpu_storage: {dtype:?}"),
    }
}
