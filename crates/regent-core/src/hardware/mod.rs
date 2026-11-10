//! What the machine can do, and how much of it Regent may use.
//!
//! Detection is behind the [`HardwareProbe`] trait so the budget rules can be
//! tested with [`testing::FakeProbe`] on any machine. GPU detection lives in the
//! `gpu` module, one implementation per operating system, and never panics.

mod budget;
mod gpu;
mod system;
pub mod testing;

use serde::Serialize;

pub use budget::{
    compute_budget, default_inference_threads, BudgetOverrides, BudgetSource, MemoryBudget,
};
pub use system::SystemProbe;

/// Installed and currently free memory, in bytes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RamInfo {
    /// All memory the OS reports, in bytes.
    pub total: u64,
    /// Memory the OS says it can hand out without swapping, in bytes.
    pub available: u64,
}

/// The processor, as far as the budget cares.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CpuInfo {
    /// The marketing name, e.g. `AMD Ryzen 7 7840U w/ Radeon 780M`.
    pub name: String,
    /// Logical processors (hyper-threads included).
    pub logical_threads: u32,
    /// Physical cores, when the platform tells us.
    pub physical_cores: Option<u32>,
}

/// What sort of GPU this is, when known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuKind {
    /// Its own VRAM.
    Dedicated,
    /// Shares system RAM.
    Integrated,
    /// Detection could not tell.
    Unknown,
}

/// One graphics adapter.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GpuInfo {
    /// The adapter name.
    pub name: String,
    /// The vendor, when known.
    pub vendor: Option<String>,
    /// Whether the adapter has its own memory.
    pub kind: GpuKind,
    /// VRAM in bytes, when known.
    pub dedicated_vram: Option<u64>,
}

/// Apple Silicon style memory shared by CPU and GPU.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnifiedMemory {
    /// The most the GPU may use, when the OS reports a limit.
    pub gpu_working_set_limit: Option<u64>,
}

/// The graphics adapters found, and whether their memory is shared.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GpuReport {
    /// Every adapter detected; may be empty.
    pub gpus: Vec<GpuInfo>,
    /// Set on machines with one memory pool for CPU and GPU.
    pub unified_memory: Option<UnifiedMemory>,
}

/// Reads the machine's capabilities.
///
/// Implementations must not panic and must be safe to call from any thread.
pub trait HardwareProbe: Send + Sync {
    /// Current memory, re-read on each call.
    fn ram(&self) -> RamInfo;
    /// The processor, read once and cached.
    fn cpu(&self) -> CpuInfo;
    /// The graphics adapters, detected once and cached.
    fn gpus(&self) -> GpuReport;
}
