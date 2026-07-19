// GPU dequantization: upload raw GGUF bytes → dequant on GPU → f16 tensor.
// Avoids CPU RAM spike from F32/F16 intermediates.

use crate::{DType, Device, Result, Shape, Storage, Tensor};
use crate::rocm_backend::RocmStorage;
use hip_runtime::memory::DeviceBuffer;
use super::GgmlDType;

/// Dequantize raw GGUF quantized bytes directly on GPU to f16.
/// Returns a f16 Tensor on the target device.
///
/// Supported: Q4_0, Q8_0. Other types fall back to CPU dequant + upload.
pub fn dequant_to_gpu(
    raw_bytes: &[u8],
    dtype: GgmlDType,
    elem_count: usize,
    device: &Device,
) -> Result<Tensor> {
    let Device::Rocm(rocm_dev) = device else {
        return Err(crate::Error::Msg(
            "dequant_to_gpu requires ROCm device".into(),
        ));
    };

    let (kernel_name, values_per_block): (&str, usize) = match dtype {
        GgmlDType::Q4_0 => ("dequant_q4_0_f16", 32),
        GgmlDType::Q8_0 => ("dequant_q8_0_f16", 32),
        _ => {
            // Unsupported quant type for GPU dequant — return error
            // Caller should fall back to CPU dequant path
            return Err(crate::Error::Msg(format!(
                "GPU dequant not supported for {dtype:?}"
            )));
        }
    };

    let num_blocks = elem_count / values_per_block;
    let out_elements = num_blocks * values_per_block;

    // Upload raw quantized bytes to GPU
    let input_buf = DeviceBuffer::from_slice(raw_bytes)
        .map_err(|e| crate::Error::Msg(format!("GPU upload quantized failed: {e}")))?;

    // Allocate output f16 buffer on GPU
    let out_buf = DeviceBuffer::<u8>::alloc(out_elements * 2)
        .map_err(|e| crate::Error::Msg(format!("GPU alloc dequant output failed: {e}")))?;

    // Launch dequant kernel
    rocm_dev.with_module("dequant", |module, _| {
        use hip_runtime::module::HipModule;
        let func = module
            .get_function(kernel_name)
            .map_err(|e| crate::Error::Msg(format!("{e}")))?;

        let block = 256u32;
        let grid = ((num_blocks as u32 + block - 1) / block).max(1);
        let num_blocks_i = num_blocks as i32;

        let input_ptr = input_buf.as_ptr() as *const std::ffi::c_void;
        let out_ptr = out_buf.as_void_ptr();

        unsafe {
            let mut params: Vec<*mut std::ffi::c_void> = vec![
                &input_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &num_blocks_i as *const _ as *mut _,
            ];
            HipModule::launch(func, (grid, 1, 1), (block, 1, 1), 0, &mut params)
                .map_err(|e| crate::Error::Msg(format!("dequant kernel failed: {e}")))?;
        }
        Ok(())
    })?;

    // Wrap GPU buffer as RocmStorage → Tensor
    let storage = RocmStorage {
        buf: out_buf,
        dtype: DType::F16,
        device: rocm_dev.clone(),
    };
    let shape = Shape::from(vec![out_elements]);
    Ok(crate::tensor::from_storage(
        Storage::Rocm(storage),
        shape,
        crate::op::BackpropOp::none(),
        false,
    ))
}
