use super::{bytes_to_cpu_storage, launch_cfg, RocmDevice, RocmStorage};
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};
use hip_runtime::memory::DeviceBuffer;
use hip_runtime::module::HipModule;

macro_rules! err {
    ($($arg:tt)*) => { crate::Error::Msg(format!($($arg)*)) };
}

impl RocmStorage {
    fn to_cpu(&self) -> Result<CpuStorage> {
        let bytes = self
            .buf
            .to_vec()
            .map_err(|e| err!("download failed: {e}"))?;
        Ok(bytes_to_cpu_storage(&bytes, self.dtype))
    }

    /// Fall back to CPU for an operation, then upload the result.
    fn cpu_fallback<F>(&self, layout: &Layout, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &Layout) -> Result<CpuStorage>,
    {
        let cpu = self.to_cpu()?;
        let result = f(&cpu, layout)?;
        self.device.storage_from_cpu_storage(&result)
    }

    /// Fall back for binary ops that need two inputs.
    fn cpu_fallback_binary<F>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        f: F,
    ) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &CpuStorage, &Layout, &Layout) -> Result<CpuStorage>,
    {
        let lhs_cpu = self.to_cpu()?;
        let rhs_cpu = rhs.to_cpu()?;
        let result = f(&lhs_cpu, &rhs_cpu, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&result)
    }
}

/// Build the permuted dims/strides info buffer for reduce kernels.
/// Reorders dimensions so non-reduced dims come first, then reduced dims.
/// Returns (info_vec, out_numel, reduce_size).
fn reduce_info(layout: &Layout, reduce_dims: &[usize]) -> (Vec<usize>, usize, usize) {
    let dims = layout.dims();
    let strides = layout.stride();
    let mut nr_dims = Vec::new();
    let mut nr_strides = Vec::new();
    let mut r_dims = Vec::new();
    let mut r_strides = Vec::new();

    for (i, (&d, &s)) in dims.iter().zip(strides.iter()).enumerate() {
        if reduce_dims.contains(&i) {
            r_dims.push(d);
            r_strides.push(s);
        } else {
            nr_dims.push(d);
            nr_strides.push(s);
        }
    }

    let out_numel: usize = nr_dims.iter().product::<usize>().max(1);
    let reduce_size: usize = r_dims.iter().product::<usize>().max(1);

    let mut info = nr_dims;
    info.extend(&r_dims);
    info.extend(&nr_strides);
    info.extend(&r_strides);

    (info, out_numel, reduce_size)
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let numel = layout.shape().elem_count();
        let byte_size = numel * self.dtype.size_in_bytes();
        let out_buf = self.device.alloc_buf(byte_size)?;

        if layout.is_contiguous() {
            let src_offset = layout.start_offset() * self.dtype.size_in_bytes();
            unsafe {
                hip_sys::hip_runtime::hipMemcpy(
                    out_buf.as_void_ptr(),
                    (self.buf.as_ptr() as *const u8).add(src_offset) as *const _,
                    byte_size,
                    hip_sys::hip_runtime::hipMemcpyKind::hipMemcpyDeviceToDevice,
                );
            }
        } else if self.dtype == DType::F32 {
            let info: Vec<usize> = [layout.dims(), layout.stride()].concat();
            let info_buf = DeviceBuffer::from_slice(&info)
                .map_err(|e| err!("info upload failed: {e}"))?;
            self.device.with_module("fill", |module, _| {
                let func = module
                    .get_function("copy_strided_f32")
                    .map_err(|e| err!("{e}"))?;
                let (grid, block) = launch_cfg(numel);
                let num_dims = layout.dims().len();
                let info_ptr = info_buf.as_ptr();
                let src_ptr = self.buf.as_ptr() as *const f32;
                let dst_ptr = out_buf.as_mut_ptr() as *mut f32;
                let src_offset = layout.start_offset();
                let dst_offset: usize = 0;
                unsafe {
                    let mut params: Vec<*mut std::ffi::c_void> = vec![
                        &src_ptr as *const _ as *mut _,
                        &dst_ptr as *const _ as *mut _,
                        &numel as *const _ as *mut _,
                        &num_dims as *const _ as *mut _,
                        &info_ptr as *const _ as *mut _,
                        &src_offset as *const _ as *mut _,
                        &dst_offset as *const _ as *mut _,
                    ];
                    HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                        .map_err(|e| err!("{e}"))?;
                }
                Ok(())
            })?;
        } else {
            return self.cpu_fallback(layout, |cpu, l| cpu.try_clone(l));
        }

        Ok(RocmStorage {
            buf: out_buf,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.to_cpu()
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return self.cpu_fallback(layout, |cpu, l| cpu.affine(l, mul, add));
        }
        let numel = layout.shape().elem_count();
        let out_buf = self.device.alloc_buf(numel * 4)?;
        let info = self.device.upload_info(layout)?;
        self.device.with_module("affine", |module, _| {
            let func = module
                .get_function("affine_f32")
                .map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let num_dims = layout.dims().len();
            let info_ptr: *const usize = info.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
            let inp_ptr =
                unsafe { (self.buf.as_ptr() as *const f32).add(layout.start_offset()) };
            let out_ptr = out_buf.as_mut_ptr() as *mut f32;
            let mul_f32 = mul as f32;
            let add_f32 = add as f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &num_dims as *const _ as *mut _,
                    &info_ptr as *const _ as *mut _,
                    &inp_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                    &mul_f32 as *const _ as *mut _,
                    &add_f32 as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return self.cpu_fallback(layout, |cpu, l| cpu.powf(l, e));
        }
        let numel = layout.shape().elem_count();
        let out_buf = self.device.alloc_buf(numel * 4)?;
        let info = self.device.upload_info(layout)?;
        self.device.with_module("unary", |module, _| {
            let func = module
                .get_function("upowf_f32")
                .map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let num_dims = layout.dims().len();
            let info_ptr: *const usize = info.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
            let inp_ptr =
                unsafe { (self.buf.as_ptr() as *const f32).add(layout.start_offset()) };
            let out_ptr = out_buf.as_mut_ptr() as *mut f32;
            let param = e as f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &num_dims as *const _ as *mut _,
                    &info_ptr as *const _ as *mut _,
                    &inp_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                    &param as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return self.cpu_fallback(layout, |cpu, l| cpu.elu(l, alpha));
        }
        let numel = layout.shape().elem_count();
        let out_buf = self.device.alloc_buf(numel * 4)?;
        let info = self.device.upload_info(layout)?;
        self.device.with_module("unary", |module, _| {
            let func = module
                .get_function("uelu_f32")
                .map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let num_dims = layout.dims().len();
            let info_ptr: *const usize = info.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
            let inp_ptr =
                unsafe { (self.buf.as_ptr() as *const f32).add(layout.start_offset()) };
            let out_ptr = out_buf.as_mut_ptr() as *mut f32;
            let param = alpha as f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &num_dims as *const _ as *mut _,
                    &info_ptr as *const _ as *mut _,
                    &inp_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                    &param as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 {
            return self.cpu_fallback(layout, |cpu, l| cpu.reduce_op(op, l, reduce_dims));
        }

        let (info_vec, out_numel, reduce_size) = reduce_info(layout, reduce_dims);
        let num_dims = layout.dims().len();
        let info_buf = DeviceBuffer::from_slice(&info_vec)
            .map_err(|e| err!("info upload failed: {e}"))?;

        let block = 256u32;
        let grid = out_numel as u32;
        let shared_bytes = block as u32 * 4; // sizeof(float) per thread

        match op {
            ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max => {
                let out_buf = self.device.alloc_buf(out_numel * 4)?;
                let kernel_name = match op {
                    ReduceOp::Sum => "fast_sum_f32",
                    ReduceOp::Max => "fast_max_f32",
                    ReduceOp::Min => "fast_min_f32",
                    _ => unreachable!(),
                };
                self.device.with_module("reduce", |module, _| {
                    let func = module.get_function(kernel_name).map_err(|e| err!("{e}"))?;
                    let info_ptr = info_buf.as_ptr();
                    let inp_ptr = unsafe {
                        (self.buf.as_ptr() as *const f32).add(layout.start_offset())
                    };
                    let out_ptr = out_buf.as_mut_ptr() as *mut f32;
                    unsafe {
                        let mut params: Vec<*mut std::ffi::c_void> = vec![
                            &inp_ptr as *const _ as *mut _,
                            &out_ptr as *const _ as *mut _,
                            &info_ptr as *const _ as *mut _,
                            &out_numel as *const _ as *mut _,
                            &reduce_size as *const _ as *mut _,
                            &num_dims as *const _ as *mut _,
                        ];
                        HipModule::launch(
                            func,
                            (grid.max(1), 1, 1),
                            (block, 1, 1),
                            shared_bytes,
                            &mut params,
                        )
                        .map_err(|e| err!("{e}"))?;
                    }
                    Ok(())
                })?;
                Ok(RocmStorage {
                    buf: out_buf,
                    dtype: DType::F32,
                    device: self.device.clone(),
                })
            }
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                let out_buf = self.device.alloc_buf(out_numel * 4)?; // u32 output
                let kernel_name = match op {
                    ReduceOp::ArgMax => "fast_argmax_f32",
                    ReduceOp::ArgMin => "fast_argmin_f32",
                    _ => unreachable!(),
                };
                // argmax/argmin need float + u32 in shared memory
                let shared_bytes = block as u32 * (4 + 4);
                self.device.with_module("reduce", |module, _| {
                    let func = module.get_function(kernel_name).map_err(|e| err!("{e}"))?;
                    let info_ptr = info_buf.as_ptr();
                    let inp_ptr = unsafe {
                        (self.buf.as_ptr() as *const f32).add(layout.start_offset())
                    };
                    let out_ptr = out_buf.as_mut_ptr() as *mut u32;
                    unsafe {
                        let mut params: Vec<*mut std::ffi::c_void> = vec![
                            &inp_ptr as *const _ as *mut _,
                            &out_ptr as *const _ as *mut _,
                            &info_ptr as *const _ as *mut _,
                            &out_numel as *const _ as *mut _,
                            &reduce_size as *const _ as *mut _,
                            &num_dims as *const _ as *mut _,
                        ];
                        HipModule::launch(
                            func,
                            (grid.max(1), 1, 1),
                            (block, 1, 1),
                            shared_bytes,
                            &mut params,
                        )
                        .map_err(|e| err!("{e}"))?;
                    }
                    Ok(())
                })?;
                Ok(RocmStorage {
                    buf: out_buf,
                    dtype: DType::U32,
                    device: self.device.clone(),
                })
            }
        }
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 {
            return self.cpu_fallback_binary(rhs, lhs_l, rhs_l, |lc, rc, ll, rl| {
                lc.cmp(op, rc, ll, rl)
            });
        }
        let numel = lhs_l.shape().elem_count();
        let out_buf = self.device.alloc_buf(numel)?; // u8 output
        let info = self.device.upload_binary_info(lhs_l, rhs_l)?;

        let kernel_name = match op {
            CmpOp::Eq => "eq_f32",
            CmpOp::Ne => "ne_f32",
            CmpOp::Lt => "lt_f32",
            CmpOp::Le => "le_f32",
            CmpOp::Gt => "gt_f32",
            CmpOp::Ge => "ge_f32",
        };

        self.device.with_module("binary", |module, _| {
            let func = module.get_function(kernel_name).map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let num_dims = lhs_l.dims().len();
            let info_ptr: *const usize = info.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
            let lhs_ptr =
                unsafe { (self.buf.as_ptr() as *const f32).add(lhs_l.start_offset()) };
            let rhs_ptr =
                unsafe { (rhs.buf.as_ptr() as *const f32).add(rhs_l.start_offset()) };
            let out_ptr = out_buf.as_mut_ptr() as *mut u8;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &num_dims as *const _ as *mut _,
                    &info_ptr as *const _ as *mut _,
                    &lhs_ptr as *const _ as *mut _,
                    &rhs_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::U8,
            device: self.device.clone(),
        })
    }

    fn to_dtype(&self, layout: &Layout, dst_dtype: DType) -> Result<Self> {
        if self.dtype == dst_dtype {
            return self.try_clone(layout);
        }

        let numel = layout.shape().elem_count();
        let kernel_name = match (self.dtype, dst_dtype) {
            (DType::F32, DType::U8) => Some("cast_f32_u8"),
            (DType::F32, DType::U32) => Some("cast_f32_u32"),
            (DType::F32, DType::I64) => Some("cast_f32_i64"),
            (DType::F32, DType::F64) => Some("cast_f32_f64"),
            (DType::U8, DType::F32) => Some("cast_u8_f32"),
            (DType::U32, DType::F32) => Some("cast_u32_f32"),
            (DType::I64, DType::F32) => Some("cast_i64_f32"),
            (DType::F64, DType::F32) => Some("cast_f64_f32"),
            (DType::U32, DType::I64) => Some("cast_u32_i64"),
            (DType::I64, DType::U32) => Some("cast_i64_u32"),
            (DType::U8, DType::U32) => Some("cast_u8_u32"),
            (DType::U32, DType::U8) => Some("cast_u32_u8"),
            _ => None,
        };

        let kernel_name = match kernel_name {
            Some(k) => k,
            None => return self.cpu_fallback(layout, |cpu, l| cpu.to_dtype(l, dst_dtype)),
        };

        let out_buf = self.device.alloc_buf(numel * dst_dtype.size_in_bytes())?;
        let info = self.device.upload_info(layout)?;

        self.device.with_module("cast", |module, _| {
            let func = module.get_function(kernel_name).map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let num_dims = layout.dims().len();
            let info_ptr: *const usize = info.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
            let inp_ptr = unsafe {
                (self.buf.as_ptr()).add(layout.start_offset() * self.dtype.size_in_bytes())
            };
            let out_ptr = out_buf.as_mut_ptr();
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &num_dims as *const _ as *mut _,
                    &info_ptr as *const _ as *mut _,
                    &inp_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: dst_dtype,
            device: self.device.clone(),
        })
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 {
            return self.cpu_fallback(layout, |cpu, l| cpu.unary_impl::<B>(l));
        }
        let func_name = format!("{}_f32", B::KERNEL);
        let numel = layout.shape().elem_count();
        let out_buf = self.device.alloc_buf(numel * 4)?;
        let info = self.device.upload_info(layout)?;

        self.device.with_module("unary", |module, _| {
            let func = module.get_function(&func_name).map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let num_dims = layout.dims().len();
            let info_ptr: *const usize = info.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
            let inp_ptr =
                unsafe { (self.buf.as_ptr() as *const f32).add(layout.start_offset()) };
            let out_ptr = out_buf.as_mut_ptr() as *mut f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &num_dims as *const _ as *mut _,
                    &info_ptr as *const _ as *mut _,
                    &inp_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return self.cpu_fallback_binary(rhs, lhs_l, rhs_l, |lc, rc, ll, rl| {
                lc.binary_impl::<B>(rc, ll, rl)
            });
        }
        let func_name = format!("{}_f32", B::KERNEL);
        let numel = lhs_l.shape().elem_count();
        let out_buf = self.device.alloc_buf(numel * 4)?;
        let info = self.device.upload_binary_info(lhs_l, rhs_l)?;

        self.device.with_module("binary", |module, _| {
            let func = module.get_function(&func_name).map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let num_dims = lhs_l.dims().len();
            let info_ptr: *const usize = info.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
            let lhs_ptr =
                unsafe { (self.buf.as_ptr() as *const f32).add(lhs_l.start_offset()) };
            let rhs_ptr =
                unsafe { (rhs.buf.as_ptr() as *const f32).add(rhs_l.start_offset()) };
            let out_ptr = out_buf.as_mut_ptr() as *mut f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &num_dims as *const _ as *mut _,
                    &info_ptr as *const _ as *mut _,
                    &lhs_ptr as *const _ as *mut _,
                    &rhs_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if t.dtype != DType::F32
            || !layout.is_contiguous()
            || !t_l.is_contiguous()
            || !f_l.is_contiguous()
        {
            let cpu_cond = self.to_cpu()?;
            let cpu_t = t.to_cpu()?;
            let cpu_f = f.to_cpu()?;
            let result = cpu_cond.where_cond(layout, &cpu_t, t_l, &cpu_f, f_l)?;
            return self.device.storage_from_cpu_storage(&result);
        }

        let numel = layout.shape().elem_count();
        let out_buf = self.device.alloc_buf(numel * 4)?;

        self.device.with_module("ternary", |module, _| {
            let func = module
                .get_function("where_cond_f32")
                .map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let num_dims: usize = 0;
            let info_ptr: *const usize = std::ptr::null();
            let cond_ptr =
                unsafe { (self.buf.as_ptr() as *const u8).add(layout.start_offset()) };
            let t_ptr = unsafe { (t.buf.as_ptr() as *const f32).add(t_l.start_offset()) };
            let f_ptr = unsafe { (f.buf.as_ptr() as *const f32).add(f_l.start_offset()) };
            let out_ptr = out_buf.as_mut_ptr() as *mut f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &num_dims as *const _ as *mut _,
                    &info_ptr as *const _ as *mut _,
                    &cond_ptr as *const _ as *mut _,
                    &t_ptr as *const _ as *mut _,
                    &f_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(err!("conv1d not supported on ROCm backend"))
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(err!("conv_transpose1d not supported on ROCm backend"))
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(err!("conv2d not supported on ROCm backend"))
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(err!("conv_transpose2d not supported on ROCm backend"))
    }

    fn avg_pool2d(
        &self,
        _: &Layout,
        _: (usize, usize),
        _: (usize, usize),
    ) -> Result<Self> {
        Err(err!("avg_pool2d not supported on ROCm backend"))
    }

    fn max_pool2d(
        &self,
        _: &Layout,
        _: (usize, usize),
        _: (usize, usize),
    ) -> Result<Self> {
        Err(err!("max_pool2d not supported on ROCm backend"))
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(err!("upsample_nearest1d not supported on ROCm backend"))
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(err!("upsample_nearest2d not supported on ROCm backend"))
    }

    fn gather(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || !layout.is_contiguous() || !ids_l.is_contiguous() {
            return self.cpu_fallback_binary(ids, layout, ids_l, |sc, ic, sl, il| {
                sc.gather(sl, ic, il, dim)
            });
        }

        let dims = layout.dims();
        let left_size: usize = dims[..dim].iter().product();
        let src_dim_size = dims[dim];
        let ids_dim_size = ids_l.dims()[dim];
        let right_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let numel = ids_l.shape().elem_count();
        let out_buf = self.device.alloc_buf(numel * 4)?;

        let kernel_name = match ids.dtype {
            DType::U32 => "gather_u32_f32",
            DType::I64 => "gather_i64_f32",
            _ => return Err(err!("gather: unsupported index type {:?}", ids.dtype)),
        };

        self.device.with_module("indexing", |module, _| {
            let func = module.get_function(kernel_name).map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let ids_ptr = unsafe {
                (ids.buf.as_ptr()).add(ids_l.start_offset() * ids.dtype.size_in_bytes())
            };
            let inp_ptr = unsafe {
                (self.buf.as_ptr() as *const f32).add(layout.start_offset())
            };
            let out_ptr = out_buf.as_mut_ptr() as *mut f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &ids_ptr as *const _ as *mut _,
                    &inp_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                    &left_size as *const _ as *mut _,
                    &src_dim_size as *const _ as *mut _,
                    &ids_dim_size as *const _ as *mut _,
                    &right_size as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn scatter_add(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32
            || !layout.is_contiguous()
            || !ids_l.is_contiguous()
            || !src_l.is_contiguous()
        {
            let cpu_self = self.to_cpu()?;
            let cpu_ids = ids.to_cpu()?;
            let cpu_src = src.to_cpu()?;
            let result = cpu_self.scatter_add(layout, &cpu_ids, ids_l, &cpu_src, src_l, dim)?;
            return self.device.storage_from_cpu_storage(&result);
        }

        // Start with a copy of self, then scatter-add into it
        let dst = self.try_clone(layout)?;
        let src_dims = src_l.dims();
        let left_size: usize = src_dims[..dim].iter().product();
        let src_dim_size = src_dims[dim];
        let dst_dim_size = layout.dims()[dim];
        let right_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let numel = src_l.shape().elem_count();

        let kernel_name = match ids.dtype {
            DType::U32 => "scatter_add_u32_f32",
            DType::I64 => "scatter_add_i64_f32",
            _ => return Err(err!("scatter_add: unsupported index type {:?}", ids.dtype)),
        };

        self.device.with_module("indexing", |module, _| {
            let func = module.get_function(kernel_name).map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let ids_ptr = unsafe {
                (ids.buf.as_ptr()).add(ids_l.start_offset() * ids.dtype.size_in_bytes())
            };
            let src_ptr = unsafe {
                (src.buf.as_ptr() as *const f32).add(src_l.start_offset())
            };
            let dst_ptr = dst.buf.as_mut_ptr() as *mut f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &ids_ptr as *const _ as *mut _,
                    &src_ptr as *const _ as *mut _,
                    &dst_ptr as *const _ as *mut _,
                    &left_size as *const _ as *mut _,
                    &src_dim_size as *const _ as *mut _,
                    &dst_dim_size as *const _ as *mut _,
                    &right_size as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(dst)
    }

    fn index_select(
        &self,
        ids: &Self,
        layout: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || !layout.is_contiguous() || !ids_l.is_contiguous() {
            return self.cpu_fallback_binary(ids, layout, ids_l, |sc, ic, sl, il| {
                sc.index_select(ic, sl, il, dim)
            });
        }

        let dims = layout.dims();
        let left_size: usize = dims[..dim].iter().product();
        let src_dim_size = dims[dim];
        let ids_dim_size = ids_l.dims()[0];
        let right_size: usize = dims[dim + 1..].iter().product::<usize>().max(1);
        let numel = left_size * ids_dim_size * right_size;
        let out_buf = self.device.alloc_buf(numel * 4)?;

        let kernel_name = match ids.dtype {
            DType::U32 => "index_select_u32_f32",
            DType::I64 => "index_select_i64_f32",
            _ => return Err(err!("index_select: unsupported index type {:?}", ids.dtype)),
        };

        self.device.with_module("indexing", |module, _| {
            let func = module.get_function(kernel_name).map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let ids_ptr = unsafe {
                (ids.buf.as_ptr()).add(ids_l.start_offset() * ids.dtype.size_in_bytes())
            };
            let inp_ptr = unsafe {
                (self.buf.as_ptr() as *const f32).add(layout.start_offset())
            };
            let out_ptr = out_buf.as_mut_ptr() as *mut f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &ids_ptr as *const _ as *mut _,
                    &inp_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                    &left_size as *const _ as *mut _,
                    &src_dim_size as *const _ as *mut _,
                    &ids_dim_size as *const _ as *mut _,
                    &right_size as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn index_add(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32
            || !layout.is_contiguous()
            || !ids_l.is_contiguous()
            || !src_l.is_contiguous()
        {
            let cpu_self = self.to_cpu()?;
            let cpu_ids = ids.to_cpu()?;
            let cpu_src = src.to_cpu()?;
            let result = cpu_self.index_add(layout, &cpu_ids, ids_l, &cpu_src, src_l, dim)?;
            return self.device.storage_from_cpu_storage(&result);
        }

        let dst = self.try_clone(layout)?;
        let src_dims = src_l.dims();
        let left_size: usize = src_dims[..dim].iter().product();
        let ids_dim_size = ids_l.dims()[0];
        let dst_dim_size = layout.dims()[dim];
        let right_size: usize = src_dims[dim + 1..].iter().product::<usize>().max(1);
        let numel = src_l.shape().elem_count();

        let kernel_name = match ids.dtype {
            DType::U32 => "index_add_u32_f32",
            DType::I64 => "index_add_i64_f32",
            _ => return Err(err!("index_add: unsupported index type {:?}", ids.dtype)),
        };

        self.device.with_module("indexing", |module, _| {
            let func = module.get_function(kernel_name).map_err(|e| err!("{e}"))?;
            let (grid, block) = launch_cfg(numel);
            let ids_ptr = unsafe {
                (ids.buf.as_ptr()).add(ids_l.start_offset() * ids.dtype.size_in_bytes())
            };
            let src_ptr = unsafe {
                (src.buf.as_ptr() as *const f32).add(src_l.start_offset())
            };
            let dst_ptr = dst.buf.as_mut_ptr() as *mut f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &numel as *const _ as *mut _,
                    &ids_ptr as *const _ as *mut _,
                    &ids_dim_size as *const _ as *mut _,
                    &src_ptr as *const _ as *mut _,
                    &dst_ptr as *const _ as *mut _,
                    &left_size as *const _ as *mut _,
                    &dst_dim_size as *const _ as *mut _,
                    &right_size as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })?;
        Ok(dst)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return self.cpu_fallback_binary(rhs, lhs_l, rhs_l, |lc, rc, ll, rl| {
                lc.matmul(rc, (b, m, n, k), ll, rl)
            });
        }

        if !lhs_l.is_contiguous() || !rhs_l.is_contiguous() {
            return self.cpu_fallback_binary(rhs, lhs_l, rhs_l, |lc, rc, ll, rl| {
                lc.matmul(rc, (b, m, n, k), ll, rl)
            });
        }

        let out_byte_size = b * m * n * 4;
        let out_buf = self.device.alloc_buf(out_byte_size)?;

        // Row-major A[m,k] * B[k,n] = C[m,n]
        // In col-major: C'[n,m] = B'[n,k] * A'[k,m]
        // sgemm(N, N, n, m, k, 1.0, B, n, A, k, 0.0, C, n)
        let lhs_ptr = unsafe {
            (self.buf.as_ptr() as *const u8)
                .add(lhs_l.start_offset() * 4) as *const std::ffi::c_void
        };
        let rhs_ptr = unsafe {
            (rhs.buf.as_ptr() as *const u8)
                .add(rhs_l.start_offset() * 4) as *const std::ffi::c_void
        };
        let out_ptr = out_buf.as_void_ptr();

        self.device.with_blas(|blas| {
            if b == 1 {
                unsafe {
                    blas.sgemm_raw(
                        false, false,
                        n, m, k,
                        1.0,
                        rhs_ptr, n,
                        lhs_ptr, k,
                        0.0,
                        out_ptr, n,
                    )
                    .map_err(|e| err!("sgemm failed: {e}"))?;
                }
            } else {
                let stride_a = (k * n) as i64; // rhs batch stride
                let stride_b = (m * k) as i64; // lhs batch stride
                let stride_c = (m * n) as i64;
                unsafe {
                    blas.sgemm_strided_batched_raw(
                        false, false,
                        n, m, k,
                        1.0,
                        rhs_ptr, n, stride_a,
                        lhs_ptr, k, stride_b,
                        0.0,
                        out_ptr, n, stride_c,
                        b,
                    )
                    .map_err(|e| err!("sgemm_strided_batched failed: {e}"))?;
                }
            }
            Ok(())
        })?;

        Ok(RocmStorage {
            buf: out_buf,
            dtype: DType::F32,
            device: self.device.clone(),
        })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, layout: &Layout) -> Result<()> {
        if self.dtype != DType::F32 {
            let cpu_self = self.to_cpu()?;
            let mut cpu_dst = dst.to_cpu()?;
            cpu_self.copy_strided_src(&mut cpu_dst, dst_offset, layout)?;
            let (bytes, _dtype) = super::cpu_storage_to_bytes(cpu_dst);
            let new_buf = DeviceBuffer::from_slice(&bytes)
                .map_err(|e| err!("upload failed: {e}"))?;
            dst.buf = new_buf;
            return Ok(());
        }

        let numel = layout.shape().elem_count();

        if layout.is_contiguous() {
            let src_offset = layout.start_offset() * 4;
            let dst_byte_offset = dst_offset * 4;
            let byte_size = numel * 4;
            unsafe {
                hip_sys::hip_runtime::hipMemcpy(
                    (dst.buf.as_mut_ptr() as *mut u8).add(dst_byte_offset) as *mut _,
                    (self.buf.as_ptr() as *const u8).add(src_offset) as *const _,
                    byte_size,
                    hip_sys::hip_runtime::hipMemcpyKind::hipMemcpyDeviceToDevice,
                );
            }
        } else {
            let info: Vec<usize> = [layout.dims(), layout.stride()].concat();
            let info_buf = DeviceBuffer::from_slice(&info)
                .map_err(|e| err!("info upload failed: {e}"))?;
            self.device.with_module("fill", |module, _| {
                let func = module
                    .get_function("copy_strided_f32")
                    .map_err(|e| err!("{e}"))?;
                let (grid, block) = launch_cfg(numel);
                let num_dims = layout.dims().len();
                let info_ptr = info_buf.as_ptr();
                let src_ptr = self.buf.as_ptr() as *const f32;
                let dst_ptr = dst.buf.as_mut_ptr() as *mut f32;
                let src_offset = layout.start_offset();
                unsafe {
                    let mut params: Vec<*mut std::ffi::c_void> = vec![
                        &src_ptr as *const _ as *mut _,
                        &dst_ptr as *const _ as *mut _,
                        &numel as *const _ as *mut _,
                        &num_dims as *const _ as *mut _,
                        &info_ptr as *const _ as *mut _,
                        &src_offset as *const _ as *mut _,
                        &dst_offset as *const _ as *mut _,
                    ];
                    HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                        .map_err(|e| err!("{e}"))?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        if self.dtype != DType::F32 {
            let cpu_self = self.to_cpu()?;
            let mut cpu_dst = dst.to_cpu()?;
            cpu_self.copy2d(&mut cpu_dst, d1, d2, src_stride1, dst_stride1, src_offset, dst_offset)?;
            let (bytes, _dtype) = super::cpu_storage_to_bytes(cpu_dst);
            let new_buf = DeviceBuffer::from_slice(&bytes)
                .map_err(|e| err!("upload failed: {e}"))?;
            dst.buf = new_buf;
            return Ok(());
        }

        self.device.with_module("fill", |module, _| {
            let func = module
                .get_function("copy2d_f32")
                .map_err(|e| err!("{e}"))?;
            let numel = d1 * d2;
            let (grid, block) = launch_cfg(numel);
            let src_ptr = self.buf.as_ptr() as *const f32;
            let dst_ptr = dst.buf.as_mut_ptr() as *mut f32;
            unsafe {
                let mut params: Vec<*mut std::ffi::c_void> = vec![
                    &src_ptr as *const _ as *mut _,
                    &dst_ptr as *const _ as *mut _,
                    &d1 as *const _ as *mut _,
                    &d2 as *const _ as *mut _,
                    &src_stride1 as *const _ as *mut _,
                    &dst_stride1 as *const _ as *mut _,
                    &src_offset as *const _ as *mut _,
                    &dst_offset as *const _ as *mut _,
                ];
                HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                    .map_err(|e| err!("{e}"))?;
            }
            Ok(())
        })
    }
}
