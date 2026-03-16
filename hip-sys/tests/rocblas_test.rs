use hip_sys::hip_runtime::*;
use hip_sys::rocblas::*;
use std::ptr;

#[test]
fn test_rocblas_create_destroy() {
    unsafe { hipSetDevice(0); }
    let mut handle: rocblas_handle = ptr::null_mut();
    let status = unsafe { rocblas_create_handle(&mut handle) };
    assert_eq!(status, ROCBLAS_STATUS_SUCCESS);
    assert!(!handle.is_null());

    let status = unsafe { rocblas_destroy_handle(handle) };
    assert_eq!(status, ROCBLAS_STATUS_SUCCESS);
}

#[test]
fn test_rocblas_sgemm() {
    unsafe { hipSetDevice(0); }

    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = A * B = [[19, 22], [43, 50]]
    // Column-major layout
    let h_a: [f32; 4] = [1.0, 3.0, 2.0, 4.0];
    let h_b: [f32; 4] = [5.0, 7.0, 6.0, 8.0];
    let expected: [f32; 4] = [19.0, 43.0, 22.0, 50.0];

    let bytes = 4 * std::mem::size_of::<f32>();
    let mut d_a: *mut std::ffi::c_void = ptr::null_mut();
    let mut d_b: *mut std::ffi::c_void = ptr::null_mut();
    let mut d_c: *mut std::ffi::c_void = ptr::null_mut();

    unsafe {
        hipMalloc(&mut d_a, bytes);
        hipMalloc(&mut d_b, bytes);
        hipMalloc(&mut d_c, bytes);
        hipMemcpy(d_a, h_a.as_ptr() as *const _, bytes, hipMemcpyKind::hipMemcpyHostToDevice);
        hipMemcpy(d_b, h_b.as_ptr() as *const _, bytes, hipMemcpyKind::hipMemcpyHostToDevice);

        let mut handle: rocblas_handle = ptr::null_mut();
        rocblas_create_handle(&mut handle);

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = rocblas_sgemm(
            handle,
            rocblas_operation::rocblas_operation_none,
            rocblas_operation::rocblas_operation_none,
            2, 2, 2,
            &alpha,
            d_a, 2,
            d_b, 2,
            &beta,
            d_c, 2,
        );
        assert_eq!(status, ROCBLAS_STATUS_SUCCESS);

        let mut result = [0.0f32; 4];
        hipMemcpy(
            result.as_mut_ptr() as *mut _,
            d_c,
            bytes,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        );

        for i in 0..4 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-4,
                "Mismatch at {i}: got {} expected {}",
                result[i],
                expected[i]
            );
        }

        rocblas_destroy_handle(handle);
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
    }
}
