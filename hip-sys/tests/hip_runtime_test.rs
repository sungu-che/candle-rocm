use hip_sys::hip_runtime::*;
use std::ptr;

#[test]
fn test_hip_get_device_count() {
    let mut count: i32 = 0;
    let err = unsafe { hipGetDeviceCount(&mut count) };
    assert_eq!(err, HIP_SUCCESS, "hipGetDeviceCount failed");
    assert!(count >= 1, "Expected at least 1 GPU, got {count}");
}

#[test]
fn test_hip_set_device() {
    let err = unsafe { hipSetDevice(0) };
    assert_eq!(err, HIP_SUCCESS, "hipSetDevice(0) failed");
}

#[test]
fn test_hip_malloc_free() {
    unsafe { hipSetDevice(0); }
    let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
    let size = 1024 * 1024; // 1MB
    let err = unsafe { hipMalloc(&mut ptr, size) };
    assert_eq!(err, HIP_SUCCESS, "hipMalloc failed");
    assert!(!ptr.is_null(), "hipMalloc returned null");

    let err = unsafe { hipFree(ptr) };
    assert_eq!(err, HIP_SUCCESS, "hipFree failed");
}

#[test]
fn test_hip_memcpy_roundtrip() {
    unsafe { hipSetDevice(0); }

    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let bytes = data.len() * std::mem::size_of::<f32>();

    let mut d_ptr: *mut std::ffi::c_void = ptr::null_mut();
    unsafe {
        assert_eq!(hipMalloc(&mut d_ptr, bytes), HIP_SUCCESS);
        assert_eq!(
            hipMemcpy(d_ptr, data.as_ptr() as *const _, bytes, hipMemcpyKind::hipMemcpyHostToDevice),
            HIP_SUCCESS
        );
    }

    let mut result = vec![0.0f32; 256];
    unsafe {
        assert_eq!(
            hipMemcpy(result.as_mut_ptr() as *mut _, d_ptr, bytes, hipMemcpyKind::hipMemcpyDeviceToHost),
            HIP_SUCCESS
        );
        hipFree(d_ptr);
    }

    assert_eq!(data, result, "H2D→D2H roundtrip mismatch");
}

#[test]
fn test_hip_device_properties() {
    unsafe { hipSetDevice(0); }
    let mut prop = std::mem::MaybeUninit::<hipDeviceProp_t>::zeroed();
    let err = unsafe { hipGetDeviceProperties(prop.as_mut_ptr(), 0) };
    assert_eq!(err, HIP_SUCCESS, "hipGetDeviceProperties failed");

    let prop = unsafe { prop.assume_init() };
    let name = unsafe { std::ffi::CStr::from_ptr(prop.name.as_ptr()) };
    let name_str = name.to_str().unwrap();
    println!("GPU: {name_str}");
    println!("VRAM: {} MB", prop.total_global_mem / (1024 * 1024));
    assert!(prop.total_global_mem > 0, "Expected non-zero VRAM");
}
