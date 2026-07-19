// Fused FP8 dequant + F16 GEMM via custom HIP kernel.
//
// Performs  Y[m, n] = scale[n] * Σ_k X[m, k] * fp8tof32(W_fp8[n, k])
//
// Avoids materializing the dequantized F16 weight tensor in VRAM:
// the kernel reads FP8 weights tile-by-tile, dequants in registers,
// and accumulates directly into the output. No intermediate F16 buffer.

use crate::{DType, Device, Result, Shape, Storage, Tensor};
use crate::rocm_backend::RocmStorage;
use hip_runtime::memory::DeviceBuffer;

/// Launch the fused FP8→F16 GEMM kernel.
///
/// # Arguments
/// * `w_fp8` — FP8 weight tensor, shape `(N, K)`, row-major, F8E4M3 dtype
/// * `x` — F16 input tensor, shape `(M, K)`, row-major
/// * `scale` — F32 per-row scale, shape `(N,)` or scalar (broadcast)
/// * `device` — ROCm device
///
/// # Returns
/// F16 output tensor, shape `(M, N)`
pub fn fp8_gemm_f16(
    w_fp8: &Tensor,
    x: &Tensor,
    scale: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let Device::Rocm(rocm_dev) = device else {
        return Err(crate::Error::Msg(
            "fp8_gemm_f16 requires ROCm device".into(),
        ));
    };

    // Validate shapes
    let w_shape = w_fp8.shape();
    let x_shape = x.shape();
    let (n, k_w) = w_shape.dims2()?;
    let (m, k_x) = x_shape.dims2()?;
    if k_w != k_x {
        return Err(crate::Error::Msg(format!(
            "fp8_gemm_f16: K mismatch: W has K={k_w}, X has K={k_x}"
        )));
    }
    let k = k_w;

    // Read scale as scalar or per-row.
    // 🚨 CRITICAL: the kernel reads scale[n] for ALL n in 0..N.
    // If scale is scalar (1 element), we MUST broadcast to N elements
    // on the CPU side — otherwise every scale[n>0] read is out-of-bounds
    // GPU memory, returning garbage and ruining the entire matmul.
    let scale_host: Vec<f32> = scale.flatten_all()?.to_vec1()?;
    let scale_len = scale_host.len();
    let scale_n = match scale_len {
        1 => {
            // Broadcast scalar to N for safe GPU reads
            vec![scale_host[0]; n]
        }
        len if len == n => scale_host,
        _ => {
            return Err(crate::Error::Msg(format!(
                "fp8_gemm_f16: scale length {scale_len} doesn't match N={n}"
            )));
        }
    };
    let scale_bytes: Vec<u8> = scale_n
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let scale_buf = DeviceBuffer::from_slice(&scale_bytes)
        .map_err(|e| crate::Error::Msg(format!("scale upload failed: {e}")))?;

    // Allocate output: (M, N) F16
    let out_elements = m * n;
    let out_bytes = out_elements * 2; // F16 = 2 bytes
    let out_buf = DeviceBuffer::<u8>::alloc(out_bytes)
        .map_err(|e| crate::Error::Msg(format!("output alloc failed: {e}")))?;

    // Extract device pointers BEFORE building param Vec.
    // storage_and_layout() borrows storage from the tensor — the temporary
    // borrow ends at this statement's semicolon. We must copy the raw
    // pointers into owned locals BEFORE the temporary drops.
    let w_ptr: *const u8 = {
        let (storage, _layout) = w_fp8.storage_and_layout();
        let Storage::Rocm(rs) = &*storage else {
            return Err(crate::Error::Msg("W must be on ROCm device".into()));
        };
        rs.buf.as_ptr() as *const u8
    };
    let x_ptr: *const u8 = {
        let (storage, _layout) = x.storage_and_layout();
        let Storage::Rocm(rs) = &*storage else {
            return Err(crate::Error::Msg("X must be on ROCm device".into()));
        };
        rs.buf.as_ptr() as *const u8
    };
    let scale_ptr = scale_buf.as_ptr() as *const f32;
    let out_ptr = out_buf.as_mut_ptr() as *mut std::ffi::c_void;

    let m_i = m as i32;
    let n_i = n as i32;
    let k_i = k as i32;

    // Launch kernel: grid = (N, M), block = 32 (1 warp)
    rocm_dev.with_module("fp8_gemm", |module, _| {
        use hip_runtime::module::HipModule;

        let func = module
            .get_function("fp8_gemm_f16")
            .map_err(|e| crate::Error::Msg(format!("get_function failed: {e}")))?;

        let block = 32u32; // 1 warp
        let grid_x = (n as u32).max(1);
        let grid_y = (m as u32).max(1);
        let grid = (grid_x, grid_y, 1u32);

        unsafe {
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &w_ptr as *const _ as *mut _,
                &x_ptr as *const _ as *mut _,
                &scale_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &m_i as *const _ as *mut _,
                &n_i as *const _ as *mut _,
                &k_i as *const _ as *mut _,
            ];
            HipModule::launch(func, grid, (block, 1, 1), 0, &mut params)
                .map_err(|e| crate::Error::Msg(format!("fp8_gemm_f16 launch failed: {e}")))?;
        }
        Ok(())
    })?;

    // Wrap output as Tensor
    let storage = RocmStorage {
        buf: out_buf,
        dtype: DType::F16,
        device: rocm_dev.clone(),
    };
    let shape = Shape::from_dims(&[m, n]);
    Ok(crate::tensor::from_storage(
        Storage::Rocm(storage),
        shape,
        crate::op::BackpropOp::none(),
        false,
    ))
}