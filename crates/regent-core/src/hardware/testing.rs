//! A [`HardwareProbe`] the tests drive by hand. It is always compiled so other
//! crates' tests can use it too.

use std::sync::{Mutex, MutexGuard};

use super::{CpuInfo, GpuReport, HardwareProbe, RamInfo};

/// A probe that answers with whatever it was built with.
#[derive(Debug)]
pub struct FakeProbe {
    ram: Mutex<RamInfo>,
    cpu: CpuInfo,
    gpus: GpuReport,
    ram_calls: Mutex<u32>,
}

impl FakeProbe {
    /// Builds a probe with fixed answers.
    pub fn new(ram: RamInfo, cpu: CpuInfo, gpus: GpuReport) -> Self {
        Self {
            ram: Mutex::new(ram),
            cpu,
            gpus,
            ram_calls: Mutex::new(0),
        }
    }

    /// Changes what [`HardwareProbe::ram`] returns from now on.
    pub fn set_ram(&self, ram: RamInfo) {
        *self.lock_ram() = ram;
    }

    /// How many times [`HardwareProbe::ram`] has been called.
    pub fn ram_calls(&self) -> u32 {
        *self
            .ram_calls
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn lock_ram(&self) -> MutexGuard<'_, RamInfo> {
        self.ram
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl HardwareProbe for FakeProbe {
    fn ram(&self) -> RamInfo {
        let mut calls = self
            .ram_calls
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *calls += 1;
        drop(calls);
        self.lock_ram().clone()
    }

    fn cpu(&self) -> CpuInfo {
        self.cpu.clone()
    }

    fn gpus(&self) -> GpuReport {
        self.gpus.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe() -> FakeProbe {
        FakeProbe::new(
            RamInfo {
                total: 16 << 30,
                available: 8 << 30,
            },
            CpuInfo {
                name: "Test CPU".to_string(),
                logical_threads: 8,
                physical_cores: Some(4),
            },
            GpuReport::default(),
        )
    }

    #[test]
    fn fake_ram_can_be_changed() {
        let probe = probe();
        assert_eq!(probe.ram().available, 8 << 30);
        probe.set_ram(RamInfo {
            total: 32 << 30,
            available: 1 << 30,
        });
        assert_eq!(probe.ram().total, 32 << 30);
        assert_eq!(probe.ram().available, 1 << 30);
    }

    #[test]
    fn fake_counts_ram_calls() {
        let probe = probe();
        assert_eq!(probe.ram_calls(), 0);
        let _ = probe.ram();
        let _ = probe.ram();
        assert_eq!(probe.ram_calls(), 2);
    }
}
