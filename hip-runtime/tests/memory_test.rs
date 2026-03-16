use hip_runtime::device::HipDevice;
use hip_runtime::memory::DeviceBuffer;

#[test]
fn test_device_buffer_alloc() {
    let _dev = HipDevice::new(0).unwrap();
    let buf = DeviceBuffer::<f32>::alloc(1024).expect("alloc failed");
    assert_eq!(buf.len(), 1024);
    assert_eq!(buf.byte_size(), 1024 * 4);
}

#[test]
fn test_device_buffer_roundtrip() {
    let _dev = HipDevice::new(0).unwrap();
    let data: Vec<f32> = (0..256).map(|i| i as f32 * 1.5).collect();
    let buf = DeviceBuffer::from_slice(&data).unwrap();
    let result = buf.to_vec().unwrap();
    assert_eq!(data, result);
}

#[test]
fn test_device_buffer_zeros() {
    let _dev = HipDevice::new(0).unwrap();
    let buf = DeviceBuffer::<f32>::alloc_zeros(128).unwrap();
    let result = buf.to_vec().unwrap();
    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn test_device_buffer_drop() {
    let _dev = HipDevice::new(0).unwrap();
    for _ in 0..100 {
        let _buf = DeviceBuffer::<f32>::alloc(1024 * 1024).unwrap();
    }
}

#[test]
fn test_device_buffer_u8() {
    let _dev = HipDevice::new(0).unwrap();
    let data: Vec<u8> = (0..255).collect();
    let buf = DeviceBuffer::from_slice(&data).unwrap();
    let result = buf.to_vec().unwrap();
    assert_eq!(data, result);
}
