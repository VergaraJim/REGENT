//! The real [`HardwareProbe`]: RAM and CPU through `sysinfo`, GPUs through
//! [`detect_gpus`](super::gpu::detect_gpus).
//!
//! Nothing here panics. `sysinfo` can fail to open the machine's system handle;
//! when it does, the probe reports zeroes and keeps working.

use std::sync::{Mutex, MutexGuard, OnceLock};

use sysinfo::System;

use super::gpu::detect_gpus;
use super::{CpuInfo, GpuReport, HardwareProbe, RamInfo};

/// Reads this machine.
pub struct SystemProbe {
    /// Opened on first use, so construction cannot fail.
    system: Mutex<Option<System>>,
    cpu: OnceLock<CpuInfo>,
    gpus: OnceLock<GpuReport>,
}

impl SystemProbe {
    /// Creates a probe. Memory is read lazily, on each call.
    pub fn new() -> Self {
        Self {
            system: Mutex::new(None),
            cpu: OnceLock::new(),
            gpus: OnceLock::new(),
        }
    }

    /// Runs `read` against the system handle, or returns `fallback` when the
    /// handle could not be opened.
    fn with_system<R>(&self, fallback: R, read: impl FnOnce(&mut System) -> R) -> R {
        let mut guard = self.lock_system();
        if guard.is_none() {
            *guard = System::new().ok();
        }
        match guard.as_mut() {
            Some(system) => read(system),
            None => fallback,
        }
    }

    fn lock_system(&self) -> MutexGuard<'_, Option<System>> {
        // A panic elsewhere must not turn every later call into one.
        self.system
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn read_ram(&self) -> RamInfo {
        self.with_system(
            RamInfo {
                total: 0,
                available: 0,
            },
            |system| {
                system.refresh_memory();
                RamInfo {
                    total: system.total_memory(),
                    available: system.available_memory(),
                }
            },
        )
    }

    fn read_cpu(&self) -> CpuInfo {
        self.with_system(
            CpuInfo {
                name: String::new(),
                logical_threads: 0,
                physical_cores: None,
            },
            |system| {
                system.refresh_cpu_all();
                let name = system
                    .cpus()
                    .first()
                    .map(|cpu| cpu.brand().trim().to_string())
                    .unwrap_or_default();
                CpuInfo {
                    name,
                    logical_threads: system.cpus().len() as u32,
                    physical_cores: System::physical_core_count().ok().map(|count| count as u32),
                }
            },
        )
    }
}

impl Default for SystemProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareProbe for SystemProbe {
    fn ram(&self) -> RamInfo {
        self.read_ram()
    }

    fn cpu(&self) -> CpuInfo {
        self.cpu.get_or_init(|| self.read_cpu()).clone()
    }

    fn gpus(&self) -> GpuReport {
        self.gpus.get_or_init(detect_gpus).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_probe_reports_installed_memory() {
        let probe = SystemProbe::new();
        assert!(probe.ram().total > 0);
    }

    #[test]
    fn cpu_is_reported_once() {
        let probe = SystemProbe::new();
        let first = probe.cpu();
        let second = probe.cpu();
        assert_eq!(first, second);
        assert!(first.logical_threads > 0);
    }

    #[test]
    fn stub_gpu_detection_finds_nothing() {
        let probe = SystemProbe::new();
        assert_eq!(probe.gpus(), GpuReport::default());
    }
}
