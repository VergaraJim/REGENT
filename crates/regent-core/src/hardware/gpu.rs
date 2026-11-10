//! GPU detection.
//!
//! A later task adds real detection. Until then Regent behaves as if the machine
//! has no GPU it may use: no adapters and no unified-memory limit.

use super::GpuReport;

/// Detects the machine's graphics adapters.
///
/// Currently a stub: always the default (empty) report.
pub(crate) fn detect_gpus() -> GpuReport {
    GpuReport::default()
}
