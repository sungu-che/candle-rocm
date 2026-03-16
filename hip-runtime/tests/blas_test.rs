use hip_runtime::blas::RocBlas;
use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;

#[test]
fn test_rocblas_gemm_identity() {
    let _dev = HipDevice::new(0).unwrap();
    let blas = RocBlas::new().unwrap();

    // I * B = B
    let a = DeviceBuffer::from_slice(&[1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let b = DeviceBuffer::from_slice(&[3.0f32, 5.0, 4.0, 6.0]).unwrap();
    let mut c = DeviceBuffer::<f32>::alloc_zeros(4).unwrap();

    blas.sgemm(false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2)
        .unwrap();

    let result = c.to_vec().unwrap();
    assert_eq!(result, vec![3.0, 5.0, 4.0, 6.0]);
}

#[test]
fn test_rocblas_gemm_known() {
    let _dev = HipDevice::new(0).unwrap();
    let blas = RocBlas::new().unwrap();

    // A(2x3) * B(3x2) = C(2x2)
    let a = DeviceBuffer::from_slice(&[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let b = DeviceBuffer::from_slice(&[7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0]).unwrap();
    let mut c = DeviceBuffer::<f32>::alloc_zeros(4).unwrap();

    blas.sgemm(false, false, 2, 2, 3, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2)
        .unwrap();

    let result = c.to_vec().unwrap();
    let expected = vec![58.0f32, 139.0, 64.0, 154.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-3, "Mismatch at {i}: got {got} expected {exp}");
    }
}

#[test]
fn test_rocblas_gemm_batched() {
    let _dev = HipDevice::new(0).unwrap();
    let blas = RocBlas::new().unwrap();

    let a = DeviceBuffer::from_slice(&[
        1.0f32, 0.0, 0.0, 1.0,
        2.0, 0.0, 0.0, 2.0,
    ]).unwrap();
    let b = DeviceBuffer::from_slice(&[
        2.0f32, 4.0, 3.0, 5.0,
        1.0, 1.0, 1.0, 1.0,
    ]).unwrap();
    let mut c = DeviceBuffer::<f32>::alloc_zeros(8).unwrap();

    blas.sgemm_strided_batched(
        false, false, 2, 2, 2,
        1.0, &a, 2, 4, &b, 2, 4,
        0.0, &mut c, 2, 4, 2,
    ).unwrap();

    let result = c.to_vec().unwrap();
    let expected = vec![2.0f32, 4.0, 3.0, 5.0, 2.0, 2.0, 2.0, 2.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-3, "Mismatch at {i}: got {got} expected {exp}");
    }
}
