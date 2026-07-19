# candle-rocm

ROCm/HIP backend for the [candle](https://github.com/huggingface/candle) ML framework. Run candle tensor operations on AMD GPUs using HIP kernels and rocBLAS.

## Requirements

### ROCm version by GPU architecture

| GPU family | Architecture | Recommended ROCm |
|------------|-------------|------------------|
| RX 6000–7000 series (RDNA2) | `gfx1030`, `gfx1031`, `gfx1032` | **7.1+** |
| RX 7900 series (RDNA3) | `gfx1100`, `gfx1101`, `gfx1102` | **7.1+** |
| MI100, MI200 (CDNA 1–2) | `gfx908`, `gfx90a` | **6.x** (e.g. 6.2.4) |
| MI200 (CDNA 2) | `gfx90a` | **6.x** (e.g. 6.2.4) |

- **RDNA2/3** (gfx1030+): ROCm 7.1+ from [rocm.com](https://rocm.docs.amd.com/en/latest/deploy/linux/install.html)
- **CDNA 1–2** (gfx908/90a): Use ROCm 6.x — install HIP SDK manually or from your distro. ROCm 7.x has limited CDNA1 support.
- Set `HIP_ARCH` to match your GPU architecture (see table above). Default is `gfx1030`.
- `hipcc` on PATH for kernel compilation.

## Usage

### Convenience crate (recommended)

```toml
[dependencies]
candle-rocm = { git = "https://github.com/vuongthai91/candle-rocm" }
```

```rust
use candle_rocm::{Device, Tensor, DType};

fn main() -> candle_rocm::Result<()> {
    let dev = Device::new_rocm(0)?;
    let a = Tensor::randn(0f32, 1., (128, 64), &dev)?;
    let b = Tensor::randn(0f32, 1., (64, 256), &dev)?;
    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}
```

### Feature flag (use with candle-core)

```toml
[dependencies]
candle-core = { git = "https://github.com/vuongthai91/candle-rocm", features = ["rocm"] }
```

```rust
use candle_core::{Device, Tensor, DType};

fn main() -> candle_core::Result<()> {
    let dev = Device::new_rocm(0)?;
    let t = Tensor::zeros((2, 3), DType::F32, &dev)?;
    Ok(())
}
```

## Device utilities

```rust
candle_rocm::is_available()       // true if a ROCm GPU is present
candle_rocm::device_count()?      // number of GPUs
candle_rocm::device_name(0)?      // e.g. "AMD Radeon RX 6900 XT"
candle_rocm::total_vram(0)?       // VRAM in bytes
```

## Supported operations

**Arithmetic / Unary**: `neg`, `abs`, `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `gelu`, `relu`, `ceil`, `floor`, `round`, `sigmoid`, `silu`, `elu`, `powf`

**Binary**: `add`, `sub`, `mul`, `div`, `min`, `max`

**Comparison**: `eq`, `ne`, `lt`, `le`, `gt`, `ge`

**Reduce**: `sum`, `max`, `min`, `argmax`, `argmin` (arbitrary dimensions)

**Matmul / GEMM**:
- `rocBLAS sgemm` (f32, batched strided)
- `rocBLAS hgemm` (f16, batched strided)
- `rocBLAS gemm_ex` (BF16 I/O, F32 compute)
- **FP8 GEMM** — fused dequant + matmul via custom HIP kernel (F8E4M3 → F16)

**Custom HIP kernels** (GPU-native, no CPU fallback):
- **RMSNorm** — F32, BF16, F16 variants (`rmsnorm_f32`, `rmsnorm_bf16`, `rmsnorm_f16`)
- **Softmax** — F32 softmax on last dim
- **Conv2D** — im2col + bias_add, F32 and F16
- **Dequant** — Q4_0, Q8_0, Q4_K, Q8_K from GGUF quantized weights → F16 on GPU
- **GroupNorm** — F32 group normalization
- **Element-wise** — unary, binary, reduce, fill (all dtypes)
- **Type casting** — all dtype pairs

**Quantized weights**: FP8 (F8E4M3), BF16, F16, F32, I8, I16, I32, I64, U8, U32

**Other**: `affine`, `where_cond`, `copy_strided`, `index_select`, `index_add`, `scatter_add`, `gather`

Operations not yet GPU-accelerated fall back to CPU automatically.

## Architecture

```
candle-rocm/
├── candle-backend/       # convenience crate (candle-rocm), re-exports candle-core + ROCm utilities
├── candle-core/          # forked candle-core with rocm feature flag
│   └── src/rocm_backend/ # RocmDevice, RocmStorage, custom op dispatch
├── hip-runtime/          # safe Rust wrappers for HIP runtime, rocBLAS, hipRAND
├── hip-sys/              # raw FFI bindings to HIP/rocBLAS/hipRAND
├── kernels/              # HIP C++ kernels (.hip files compiled to .hsaco)
└── poc/                  # proof-of-concept examples
```

### GPU kernel flow

Custom ops are dispatched from `RocmStorage` to GPU kernels via `custom_op1_gpu` / `custom_op2_gpu`. When no GPU kernel exists for an op, execution falls back to `CpuStorage` (CPU) automatically.

```
Tensor → Storage::Rocm(RocmStorage)
  → custom_op1_gpu / custom_op2_gpu → HipModule::launch(kernel)
  → fallback: to_cpu() → CpuStorage op → DeviceBuffer::from_slice upload
```

## License

MIT