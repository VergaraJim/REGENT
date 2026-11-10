//! GPU detection, dispatched per operating system.
//!
//! Detection must never stop the app from launching or bring it down: a missing
//! driver, DLL or framework yields an empty or partial report. Nothing here may
//! panic, and no GPU library is linked statically.

use super::GpuReport;

mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

/// Detects the machine's graphics adapters.
#[cfg(target_os = "windows")]
pub(crate) fn detect_gpus() -> GpuReport {
    windows::detect()
}

/// Detects the machine's graphics adapters.
#[cfg(target_os = "linux")]
pub(crate) fn detect_gpus() -> GpuReport {
    linux::detect()
}

/// Detects the machine's graphics adapters.
#[cfg(target_os = "macos")]
pub(crate) fn detect_gpus() -> GpuReport {
    macos::detect()
}

/// Detects the machine's graphics adapters.
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
pub(crate) fn detect_gpus() -> GpuReport {
    GpuReport::default()
}

/// The vendor name for a PCI vendor ID.
///
/// Unknown vendors keep their ID in hex, so the report never invents a name.
#[cfg_attr(not(any(target_os = "windows", target_os = "linux")), allow(dead_code))]
pub(crate) fn vendor_name(vendor_id: u32) -> String {
    match vendor_id {
        0x10DE => "NVIDIA".to_string(),
        0x1002 => "AMD".to_string(),
        0x8086 => "Intel".to_string(),
        other => format!("{other:#06X}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vendors_get_names() {
        assert_eq!(vendor_name(0x10DE), "NVIDIA");
        assert_eq!(vendor_name(0x1002), "AMD");
        assert_eq!(vendor_name(0x8086), "Intel");
    }

    #[test]
    fn unknown_vendors_keep_their_hex_id() {
        assert_eq!(vendor_name(0x1234), "0x1234");
    }
}
