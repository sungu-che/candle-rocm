use hip_runtime::device::HipDevice;

#[test]
fn test_device_new() {
    let dev = HipDevice::new(0).expect("Failed to create device");
    assert_eq!(dev.ordinal(), 0);
}

#[test]
fn test_device_count() {
    let count = HipDevice::device_count().expect("Failed to get device count");
    assert!(count >= 1);
}

#[test]
fn test_device_name() {
    let dev = HipDevice::new(0).unwrap();
    let name = dev.name().unwrap();
    println!("Device name: {name}");
    assert!(!name.is_empty());
}

#[test]
fn test_device_memory() {
    let dev = HipDevice::new(0).unwrap();
    let mem = dev.total_memory().unwrap();
    println!("Device memory: {} MB", mem / (1024 * 1024));
    assert!(mem > 1024 * 1024 * 1024, "Expected at least 1GB VRAM");
}

#[test]
fn test_device_synchronize() {
    let dev = HipDevice::new(0).unwrap();
    dev.synchronize().expect("Synchronize failed");
}
