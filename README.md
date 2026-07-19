# candle-rocm

ROCm/HIP backend for the [candle](https://github.com/huggingface/candle) ML framework. Run candle tensor operations on AMD GPUs.

## Requirements

- AMD GPU with ROCm support
- ROCm 5.x+ with HIP runtime and rocBLAS
- `hipcc` on PATH (for kernel compilation)
- Set `HIP_ARCH` to your GPU architecture (e.g. `gfx1010` for RX 5700 XT, `gfx1030` for RX 6900 XT, `gfx1100` for RX 7900 XTX)

## Usage

### Convenience crate (recommended)

```toml
[dependencies]
candle-rocm = { git = "https://github.com/vuongthai91/candle-rocm" }
```

```rust
use candle_rocm::{Tensor, DType};

fn main() -> candle_rocm::Result<()> {
    let dev = candle_rocm::device(0)?;
    let a = Tensor::randn(0f32, 1., (128, 64), &dev)?;
    let b = Tensor::randn(0f32, 1., (64, 256), &dev)?;
    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}
```

### Feature flag

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
candle_rocm::device_name(0)?      // e.g. "AMD Radeon RX 5700 XT"
candle_rocm::total_vram(0)?       // VRAM in bytes
```

## Supported operations

- **Unary**: neg, abs, exp, log, sin, cos, tanh, sqrt, gelu, relu, ceil, floor, round, sigmoid, silu, and more
- **Binary**: add, sub, mul, div, min, max
- **Comparison**: eq, ne, lt, le, gt, ge
- **Reduce**: sum, max, min, argmax, argmin (arbitrary dimensions)
- **Matmul**: rocBLAS sgemm (f32), CPU fallback for other dtypes
- **Type casting**: all dtype pairs via HIP kernels
- **Indexing**: gather, scatter_add, index_select, index_add
- **Other**: affine, where_cond, copy_strided, powf, elu

Operations not yet GPU-accelerated (conv, pool, upsample) fall back to CPU automatically.

## Architecture

```
candle-rocm/
├── candle-backend/   # convenience crate (candle-rocm), re-exports candle-core + ROCm utilities
├── candle-core/      # forked candle-core with rocm feature flag
├── hip-runtime/      # safe Rust wrappers for HIP runtime, rocBLAS, hipRAND
├── hip-sys/          # raw FFI bindings to HIP/rocBLAS/hipRAND
├── kernels/          # HIP C++ kernels (.hip files)
└── poc/              # proof-of-concept examples
```

## License

MIT
