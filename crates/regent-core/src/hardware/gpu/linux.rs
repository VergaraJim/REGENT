//! Linux GPU detection: sysfs for every vendor, NVML for NVIDIA.
//!
//! The sysfs walk is compiled on every OS so its tests run everywhere; only
//! [`detect`] and the NVML pass are Linux-only. `nvml-wrapper` loads
//! `libnvidia-ml.so.1` at runtime, so a machine without the driver simply has no
//! NVIDIA entries instead of a failed launch.

use std::fs;
use std::path::Path;

use super::vendor_name;
use crate::hardware::{GpuInfo, GpuKind};

const AMD_VENDOR_ID: u32 = 0x1002;
const NVIDIA_VENDOR_ID: u32 = 0x10DE;

/// Below this, an AMD "VRAM" figure is an APU carve-out from system RAM, not
/// memory of its own. Such a GPU is treated as integrated and gets no VRAM.
const AMD_APU_CARVEOUT_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

/// Detects the machine's graphics adapters.
#[cfg(target_os = "linux")]
pub(super) fn detect() -> super::GpuReport {
    let mut gpus = detect_sysfs(Path::new("/sys/class/drm"));
    gpus.extend(detect_nvml());
    super::GpuReport {
        gpus,
        unified_memory: None,
    }
}

/// Reads the `card<N>` entries under a DRM sysfs root.
///
/// Connector entries such as `card0-HDMI-A-1` are ignored, as are entries whose
/// vendor cannot be read: a figure is never guessed.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn detect_sysfs(drm_root: &Path) -> Vec<GpuInfo> {
    let Ok(entries) = fs::read_dir(drm_root) else {
        return Vec::new();
    };
    let mut gpus: Vec<GpuInfo> = entries
        .flatten()
        .filter_map(|entry| read_card(&entry.path()))
        .collect();
    // `read_dir` order is unspecified; sorting keeps reports stable between runs.
    gpus.sort_by(|a, b| a.name.cmp(&b.name));
    gpus
}

fn read_card(card: &Path) -> Option<GpuInfo> {
    let card_name = card.file_name()?.to_str()?;
    let number = card_name.strip_prefix("card")?;
    if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }

    let device = card.join("device");
    let vendor_id = read_hex(&device.join("vendor"))?;
    let vendor = vendor_name(vendor_id);
    let name = format!("{vendor} GPU ({card_name})");

    match vendor_id {
        // NVML names these properly, so sysfs would only add a duplicate.
        NVIDIA_VENDOR_ID => None,
        AMD_VENDOR_ID => match read_u64(&device.join("mem_info_vram_total")) {
            Some(bytes) if bytes >= AMD_APU_CARVEOUT_LIMIT => Some(GpuInfo {
                name,
                vendor: Some(vendor),
                kind: GpuKind::Dedicated,
                dedicated_vram: Some(bytes),
            }),
            _ => Some(GpuInfo {
                name,
                vendor: Some(vendor),
                kind: GpuKind::Integrated,
                dedicated_vram: None,
            }),
        },
        // Integrated or unknown: no VRAM, because none can be established.
        _ => Some(GpuInfo {
            name,
            vendor: Some(vendor),
            kind: GpuKind::Unknown,
            dedicated_vram: None,
        }),
    }
}

/// Reads a hex number such as the contents of `device/vendor`.
fn read_hex(path: &Path) -> Option<u32> {
    let text = fs::read_to_string(path).ok()?;
    let digits = text.trim();
    let digits = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
        .unwrap_or(digits);
    u32::from_str_radix(digits, 16).ok()
}

/// Reads a decimal byte count such as `device/mem_info_vram_total`.
fn read_u64(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Lists NVIDIA adapters through NVML, or nothing when it is unavailable.
#[cfg(target_os = "linux")]
fn detect_nvml() -> Vec<GpuInfo> {
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        return Vec::new();
    };
    let Ok(count) = nvml.device_count() else {
        return Vec::new();
    };

    let mut gpus = Vec::new();
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        let Ok(name) = device.name() else {
            continue;
        };
        gpus.push(GpuInfo {
            name,
            vendor: Some(vendor_name(NVIDIA_VENDOR_ID)),
            kind: GpuKind::Dedicated,
            dedicated_vram: device.memory_info().ok().map(|memory| memory.total),
        });
    }
    gpus
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Builds a fake DRM tree and returns its root.
    fn fake_drm(cards: &[(&str, Option<u32>, Option<u64>)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("temp dir");
        for (name, vendor, vram) in cards {
            let device = root.path().join(name).join("device");
            if let Some(vendor) = vendor {
                fs::create_dir_all(&device).expect("device dir");
                fs::write(device.join("vendor"), format!("0x{vendor:04x}\n")).expect("vendor");
            }
            if let Some(vram) = vram {
                fs::write(device.join("mem_info_vram_total"), format!("{vram}\n")).expect("vram");
            }
        }
        root
    }

    #[test]
    fn amd_with_dedicated_memory_is_dedicated() {
        let drm = fake_drm(&[("card0", Some(0x1002), Some(8 * GIB))]);
        let gpus = detect_sysfs(drm.path());
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].kind, GpuKind::Dedicated);
        assert_eq!(gpus[0].dedicated_vram, Some(8 * GIB));
        assert_eq!(gpus[0].vendor.as_deref(), Some("AMD"));
    }

    #[test]
    fn amd_with_a_small_carve_out_is_integrated() {
        let drm = fake_drm(&[("card0", Some(0x1002), Some(512 * 1024 * 1024))]);
        let gpus = detect_sysfs(drm.path());
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].kind, GpuKind::Integrated);
        assert_eq!(gpus[0].dedicated_vram, None);
    }

    #[test]
    fn amd_without_a_vram_figure_is_integrated() {
        let drm = fake_drm(&[("card0", Some(0x1002), None)]);
        let gpus = detect_sysfs(drm.path());
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].kind, GpuKind::Integrated);
        assert_eq!(gpus[0].dedicated_vram, None);
    }

    #[test]
    fn nvidia_sysfs_entries_are_skipped() {
        let drm = fake_drm(&[("card0", Some(0x10DE), Some(8 * GIB))]);
        assert!(detect_sysfs(drm.path()).is_empty());
    }

    #[test]
    fn other_vendors_are_unknown_without_vram() {
        let drm = fake_drm(&[("card0", Some(0x8086), None)]);
        let gpus = detect_sysfs(drm.path());
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].kind, GpuKind::Unknown);
        assert_eq!(gpus[0].dedicated_vram, None);
    }

    #[test]
    fn connector_entries_are_ignored() {
        let drm = fake_drm(&[
            ("card0", Some(0x1002), Some(8 * GIB)),
            ("card0-HDMI-A-1", Some(0x1002), Some(8 * GIB)),
            ("card1-DP-2", None, None),
        ]);
        let gpus = detect_sysfs(drm.path());
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].kind, GpuKind::Dedicated);
    }

    #[test]
    fn card_entries_without_a_suffix_are_ignored() {
        let drm = fake_drm(&[
            ("card", Some(0x1002), Some(8 * GIB)),
            ("renderD128", Some(0x1002), None),
        ]);
        assert!(detect_sysfs(drm.path()).is_empty());
    }

    #[test]
    fn missing_files_and_roots_cause_no_panic() {
        let drm = tempfile::tempdir().expect("temp dir");
        fs::create_dir_all(drm.path().join("card0")).expect("card dir");
        assert!(detect_sysfs(drm.path()).is_empty());
        assert!(detect_sysfs(&drm.path().join("does-not-exist")).is_empty());
    }
}
