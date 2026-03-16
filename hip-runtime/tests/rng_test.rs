use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;
use hip_runtime::rng::HipRng;
use std::path::Path;

fn kernel_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels")
}

#[test]
fn test_rng_uniform() {
    let _dev = HipDevice::new(0).unwrap();
    let mut rng = HipRng::new(42, &kernel_dir()).unwrap();
    let mut buf = DeviceBuffer::<f32>::alloc(1024).unwrap();
    rng.uniform_f32(&mut buf).unwrap();
    let vals = buf.to_vec().unwrap();
    assert!(vals.iter().all(|&v| v > 0.0 && v <= 1.0));
}

#[test]
fn test_rng_normal() {
    let _dev = HipDevice::new(0).unwrap();
    let mut rng = HipRng::new(42, &kernel_dir()).unwrap();
    let mut buf = DeviceBuffer::<f32>::alloc(4096).unwrap();
    rng.normal_f32(&mut buf, 0.0, 1.0).unwrap();
    let vals = buf.to_vec().unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(mean.abs() < 0.1, "Mean {mean} too far from 0");
}

#[test]
fn test_rng_reproducibility() {
    let _dev = HipDevice::new(0).unwrap();

    let gen = |seed| {
        let mut rng = HipRng::new(seed, &kernel_dir()).unwrap();
        let mut buf = DeviceBuffer::<f32>::alloc(256).unwrap();
        rng.uniform_f32(&mut buf).unwrap();
        buf.to_vec().unwrap()
    };

    assert_eq!(gen(123), gen(123));
    assert_ne!(gen(123), gen(456));
}
