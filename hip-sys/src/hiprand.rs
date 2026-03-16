//! hipRAND FFI bindings.

use std::ffi::c_void;
use std::os::raw::c_int;

pub type hiprandGenerator_t = *mut c_void;
pub type hiprandStatus_t = c_int;

pub const HIPRAND_STATUS_SUCCESS: hiprandStatus_t = 0;

#[repr(C)]
pub enum hiprandRngType_t {
    HIPRAND_RNG_PSEUDO_DEFAULT = 400,
    HIPRAND_RNG_PSEUDO_XORWOW = 401,
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 408,
}

extern "C" {
    pub fn hiprandCreateGenerator(
        generator: *mut hiprandGenerator_t,
        rng_type: hiprandRngType_t,
    ) -> hiprandStatus_t;

    pub fn hiprandDestroyGenerator(generator: hiprandGenerator_t) -> hiprandStatus_t;

    pub fn hiprandSetPseudoRandomGeneratorSeed(
        generator: hiprandGenerator_t,
        seed: u64,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateUniform(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateNormal(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateUniformDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateNormalDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> hiprandStatus_t;
}
