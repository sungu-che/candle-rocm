#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct RocmDevice;

#[derive(Debug)]
pub struct RocmStorage;

macro_rules! fail {
    () => {
        unimplemented!("rocm support has not been enabled, add `rocm` feature to enable.")
    };
}

impl crate::backend::BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn conv1d(&self, _: &Layout, _: &Self, _: &Layout, _: &crate::conv::ParamsConv1D) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn conv_transpose1d(&self, _: &Layout, _: &Self, _: &Layout, _: &crate::conv::ParamsConvTranspose1D) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn conv2d(&self, _: &Layout, _: &Self, _: &Layout, _: &crate::conv::ParamsConv2D) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn conv_transpose2d(&self, _: &Layout, _: &Self, _: &Layout, _: &crate::conv::ParamsConvTranspose2D) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn scatter_add(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn index_add(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn matmul(&self, _: &Self, _: (usize, usize, usize, usize), _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn copy2d(&self, _: &mut Self, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

impl crate::backend::BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }

    fn zeros_impl(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn ones_impl(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    unsafe fn alloc_uninit(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
