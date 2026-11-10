//! macOS GPU detection through Metal.
//!
//! Apple Silicon has one memory pool: its GPU is listed as integrated and the
//! Metal working-set limit becomes the unified-memory figure. Intel Macs report
//! a non-unified device, whose working-set limit stands in for dedicated VRAM.

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use crate::hardware::{GpuInfo, GpuKind, GpuReport, UnifiedMemory};

/// Detects the machine's graphics adapters.
pub(super) fn detect() -> GpuReport {
    // Metal's factory is safe to call from any thread and only returns `None`
    // when the machine has no GPU to report.
    let Some(device) = MTLCreateSystemDefaultDevice() else {
        return GpuReport::default();
    };

    let limit = device.recommendedMaxWorkingSetSize();
    let gpu = GpuInfo {
        name: "Apple GPU".to_string(),
        vendor: Some("Apple".to_string()),
        kind: GpuKind::Integrated,
        dedicated_vram: None,
    };

    if device.hasUnifiedMemory() {
        return GpuReport {
            gpus: vec![gpu],
            unified_memory: Some(UnifiedMemory {
                gpu_working_set_limit: Some(limit),
            }),
        };
    }

    GpuReport {
        gpus: vec![GpuInfo {
            kind: GpuKind::Dedicated,
            dedicated_vram: Some(limit),
            ..gpu
        }],
        unified_memory: None,
    }
}
