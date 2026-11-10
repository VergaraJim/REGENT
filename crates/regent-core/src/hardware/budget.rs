//! How much memory Regent may plan on using.
//!
//! Detection is only ever a starting point: a user override always wins, and an
//! override that is not a finite positive number is ignored.

use serde::Serialize;

use super::{CpuInfo, GpuKind, GpuReport, RamInfo};

/// Bytes in a gigabyte as the budget means it: 1024³.
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Manual overrides, in GB (1 GB = 1024³ bytes).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct BudgetOverrides {
    /// Replaces the RAM budget, in GB.
    pub ram_gb: Option<f64>,
    /// Replaces the VRAM budget, in GB.
    pub vram_gb: Option<f64>,
}

/// Where a budget figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetSource {
    /// The user set it.
    Override,
    /// A probe read it from the machine.
    Detected,
    /// Nothing was readable, so installed RAM stands in.
    InstalledFallback,
    /// Nothing usable was found.
    None,
}

/// What the memory forecast may assume, in bytes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryBudget {
    /// The RAM pool, in bytes.
    pub ram: u64,
    /// Where `ram` came from.
    pub ram_source: BudgetSource,
    /// The accelerator memory pool, in bytes.
    pub vram: u64,
    /// Where `vram` came from.
    pub vram_source: BudgetSource,
    /// Whether CPU and GPU share one memory pool.
    pub unified: bool,
}

/// Turns machine facts and overrides into the budget the forecast uses.
///
/// On unified-memory machines the VRAM pool is carved out of the RAM pool, so
/// the estimator must also count GPU-offloaded bytes against RAM.
pub fn compute_budget(
    ram: &RamInfo,
    gpus: &GpuReport,
    overrides: &BudgetOverrides,
) -> MemoryBudget {
    let (ram_bytes, ram_source) = budget_ram(ram, overrides);
    let (vram, vram_source, unified) = match &gpus.unified_memory {
        Some(memory) => {
            let (vram, source) = match override_bytes(overrides.vram_gb) {
                Some(bytes) => (bytes, BudgetSource::Override),
                None => match memory.gpu_working_set_limit {
                    Some(limit) if limit > 0 => (limit, BudgetSource::Detected),
                    _ => (0, BudgetSource::None),
                },
            };
            (vram, source, true)
        }
        None => {
            let (vram, source) = match override_bytes(overrides.vram_gb) {
                Some(bytes) => (bytes, BudgetSource::Override),
                None => match largest_dedicated_vram(gpus) {
                    Some(bytes) => (bytes, BudgetSource::Detected),
                    None => (0, BudgetSource::None),
                },
            };
            (vram, source, false)
        }
    };

    MemoryBudget {
        ram: ram_bytes,
        ram_source,
        vram,
        vram_source,
        unified,
    }
}

/// The number of threads the text model should use by default.
///
/// One core is left for the rest of the app, never dropping below one thread.
pub fn default_inference_threads(cpu: &CpuInfo) -> u32 {
    match cpu.physical_cores {
        Some(cores) => cores.saturating_sub(1).max(1),
        None => (cpu.logical_threads / 2).saturating_sub(1).max(1),
    }
}

fn budget_ram(ram: &RamInfo, overrides: &BudgetOverrides) -> (u64, BudgetSource) {
    if let Some(bytes) = override_bytes(overrides.ram_gb) {
        return (bytes, BudgetSource::Override);
    }
    if ram.available > 0 {
        return (ram.available, BudgetSource::Detected);
    }
    if ram.total > 0 {
        return (ram.total, BudgetSource::InstalledFallback);
    }
    (0, BudgetSource::None)
}

fn largest_dedicated_vram(gpus: &GpuReport) -> Option<u64> {
    gpus.gpus
        .iter()
        .filter(|gpu| gpu.kind == GpuKind::Dedicated)
        .filter_map(|gpu| gpu.dedicated_vram)
        .max()
        .filter(|bytes| *bytes > 0)
}

/// GB to bytes, or `None` when the override is unusable.
fn override_bytes(gb: Option<f64>) -> Option<u64> {
    match gb {
        Some(value) if value.is_finite() && value > 0.0 => Some((value * GIB).floor() as u64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuInfo, UnifiedMemory};

    fn ram(total: u64, available: u64) -> RamInfo {
        RamInfo { total, available }
    }

    fn gpu(kind: GpuKind, vram: Option<u64>) -> GpuInfo {
        GpuInfo {
            name: "Test GPU".to_string(),
            vendor: None,
            kind,
            dedicated_vram: vram,
        }
    }

    fn dedicated(vram: u64) -> GpuReport {
        GpuReport {
            gpus: vec![gpu(GpuKind::Dedicated, Some(vram))],
            unified_memory: None,
        }
    }

    fn unified(limit: Option<u64>) -> GpuReport {
        GpuReport {
            gpus: vec![gpu(GpuKind::Integrated, None)],
            unified_memory: Some(UnifiedMemory {
                gpu_working_set_limit: limit,
            }),
        }
    }

    #[test]
    fn ram_override_wins() {
        let over = BudgetOverrides {
            ram_gb: Some(2.0),
            vram_gb: None,
        };
        let budget = compute_budget(&ram(64 << 30, 32 << 30), &GpuReport::default(), &over);
        assert_eq!(budget.ram, 2 * 1024 * 1024 * 1024);
        assert_eq!(budget.ram_source, BudgetSource::Override);
    }

    #[test]
    fn ram_uses_available_when_detected() {
        let budget = compute_budget(
            &ram(64 << 30, 32 << 30),
            &GpuReport::default(),
            &BudgetOverrides::default(),
        );
        assert_eq!(budget.ram, 32 << 30);
        assert_eq!(budget.ram_source, BudgetSource::Detected);
    }

    #[test]
    fn ram_falls_back_to_total_when_nothing_is_available() {
        let budget = compute_budget(
            &ram(16 << 30, 0),
            &GpuReport::default(),
            &BudgetOverrides::default(),
        );
        assert_eq!(budget.ram, 16 << 30);
        assert_eq!(budget.ram_source, BudgetSource::InstalledFallback);
    }

    #[test]
    fn ram_override_must_be_finite_positive() {
        for bad in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            let over = BudgetOverrides {
                ram_gb: Some(bad),
                vram_gb: None,
            };
            let budget = compute_budget(&ram(16 << 30, 8 << 30), &GpuReport::default(), &over);
            assert_eq!(budget.ram, 8 << 30, "override {bad} should be ignored");
            assert_eq!(budget.ram_source, BudgetSource::Detected);
        }
    }

    #[test]
    fn vram_takes_the_largest_dedicated_gpu() {
        let gpus = GpuReport {
            gpus: vec![
                gpu(GpuKind::Dedicated, Some(4 << 30)),
                gpu(GpuKind::Dedicated, Some(8 << 30)),
            ],
            unified_memory: None,
        };
        let budget = compute_budget(&ram(64 << 30, 32 << 30), &gpus, &BudgetOverrides::default());
        assert_eq!(budget.vram, 8 << 30);
        assert_eq!(budget.vram_source, BudgetSource::Detected);
        assert!(!budget.unified);
    }

    #[test]
    fn integrated_gpu_never_counts() {
        let gpus = GpuReport {
            gpus: vec![gpu(GpuKind::Integrated, Some(8 << 30))],
            unified_memory: None,
        };
        let budget = compute_budget(&ram(64 << 30, 32 << 30), &gpus, &BudgetOverrides::default());
        assert_eq!(budget.vram, 0);
        assert_eq!(budget.vram_source, BudgetSource::None);
    }

    #[test]
    fn unknown_gpu_never_counts() {
        let gpus = GpuReport {
            gpus: vec![gpu(GpuKind::Unknown, Some(8 << 30))],
            unified_memory: None,
        };
        let budget = compute_budget(&ram(64 << 30, 32 << 30), &gpus, &BudgetOverrides::default());
        assert_eq!(budget.vram, 0);
        assert_eq!(budget.vram_source, BudgetSource::None);
    }

    #[test]
    fn vram_override_wins() {
        let over = BudgetOverrides {
            ram_gb: None,
            vram_gb: Some(3.0),
        };
        let budget = compute_budget(&ram(64 << 30, 32 << 30), &dedicated(8 << 30), &over);
        assert_eq!(budget.vram, 3 * 1024 * 1024 * 1024);
        assert_eq!(budget.vram_source, BudgetSource::Override);
    }

    #[test]
    fn unified_uses_the_working_set_limit() {
        let budget = compute_budget(
            &ram(32 << 30, 16 << 30),
            &unified(Some(10 << 30)),
            &BudgetOverrides::default(),
        );
        assert!(budget.unified);
        assert_eq!(budget.vram, 10 << 30);
        assert_eq!(budget.vram_source, BudgetSource::Detected);
    }

    #[test]
    fn unified_without_a_limit_has_no_vram() {
        let budget = compute_budget(
            &ram(32 << 30, 16 << 30),
            &unified(None),
            &BudgetOverrides::default(),
        );
        assert!(budget.unified);
        assert_eq!(budget.vram, 0);
        assert_eq!(budget.vram_source, BudgetSource::None);
    }

    #[test]
    fn unified_ignores_dedicated_vram() {
        let gpus = GpuReport {
            gpus: vec![gpu(GpuKind::Dedicated, Some(8 << 30))],
            unified_memory: Some(UnifiedMemory {
                gpu_working_set_limit: None,
            }),
        };
        let budget = compute_budget(&ram(32 << 30, 16 << 30), &gpus, &BudgetOverrides::default());
        assert_eq!(budget.vram, 0);
        assert_eq!(budget.vram_source, BudgetSource::None);
    }

    fn cpu(physical: Option<u32>, logical: u32) -> CpuInfo {
        CpuInfo {
            name: "Test CPU".to_string(),
            logical_threads: logical,
            physical_cores: physical,
        }
    }

    #[test]
    fn threads_leave_one_core_free() {
        assert_eq!(default_inference_threads(&cpu(Some(8), 16)), 7);
    }

    #[test]
    fn threads_never_drop_below_one() {
        assert_eq!(default_inference_threads(&cpu(Some(1), 2)), 1);
        assert_eq!(default_inference_threads(&cpu(Some(0), 2)), 1);
    }

    #[test]
    fn threads_fall_back_to_half_the_logical_count() {
        assert_eq!(default_inference_threads(&cpu(None, 16)), 7);
        assert_eq!(default_inference_threads(&cpu(None, 1)), 1);
    }
}
