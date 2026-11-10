//! Builds the full memory forecast from settings, hardware and model files, and
//! runs it on a worker thread that only ever computes the newest request.
//!
//! Everything here is pure logic over plain data: the [`HardwareProbe`] and
//! [`ModelInspector`] traits are the only ways to touch the machine, so the
//! forecast is unit tested with fake hardware and fake model files.
//!
//! Two rules keep the forecast honest:
//! - every model file is inspected afresh when the estimate is computed, but
//!   the inspector caches by `(path, size, mtime)` so unchanged files are only
//!   read once;
//! - a file that can't be read becomes a row with an error and zero bytes, not
//!   a silently missing model.

use std::collections::HashMap;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::SystemTime;

use crate::context::{
    clamp_requested, compaction_thresholds, context_limits, system_max_context, usable_memory,
    ClampReason, ClampedContext, CompactionThresholds, ContextLimits, ENGINE_MAX_CONTEXT,
};
use crate::footprint::{
    combined_status, model_footprint, FitStatus, Footprint, KvCacheType, KvEstimate, ModelInput,
    ModelKind, FIXED_EXTRAS_BYTES,
};
use crate::gguf::{read_gguf_meta, ModelMeta};
use crate::hardware::{
    compute_budget, BudgetOverrides, CpuInfo, GpuInfo, HardwareProbe, MemoryBudget, RamInfo,
};
use crate::settings::Settings;

/// The most entries the inspector cache keeps before it is cleared.
const CACHE_LIMIT: usize = 32;

/// What one model file looks like, as far as the forecast is concerned.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct FileInfo {
    /// The size on disk, in bytes.
    pub size: u64,
    /// When the file was last modified, when the OS reports it.
    pub modified: Option<SystemTime>,
    /// The GGUF metadata, when the file carries a readable header.
    pub meta: Option<ModelMeta>,
}

/// Reads what the forecast needs to know about a model file.
///
/// Implementations must be safe to call from any thread.
pub trait ModelInspector: Send + Sync {
    /// Reads `path`'s size, modification time and, for GGUF files, metadata.
    fn inspect(&self, path: &Path) -> Result<FileInfo, String>;
}

/// The real [`ModelInspector`]: reads files from disk and caches what it saw.
pub struct FsInspector {
    /// Keyed on path, size and modification time, so a changed file is re-read.
    cache: Mutex<HashMap<CacheKey, FileInfo>>,
}

/// A cache lookup key: the same file at the same size and mtime is unchanged.
type CacheKey = (PathBuf, u64, Option<SystemTime>);

impl FsInspector {
    /// Creates an inspector with an empty cache.
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn lock_cache(&self) -> MutexGuard<'_, HashMap<CacheKey, FileInfo>> {
        self.cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Default for FsInspector {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelInspector for FsInspector {
    fn inspect(&self, path: &Path) -> Result<FileInfo, String> {
        let metadata = std::fs::metadata(path).map_err(read_error)?;
        let size = metadata.len();
        let modified = metadata.modified().ok();
        let key = (path.to_path_buf(), size, modified);

        if let Some(info) = self.lock_cache().get(&key) {
            return Ok(info.clone());
        }

        let info = read_file_info(path, size, modified)?;
        let mut cache = self.lock_cache();
        if cache.len() >= CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, info.clone());
        Ok(info)
    }
}

/// The user-facing error for a file that can't be read.
fn read_error(error: std::io::Error) -> String {
    if error.kind() == ErrorKind::NotFound {
        "File not found".to_string()
    } else {
        "Couldn't read this file".to_string()
    }
}

/// Reads a file's size, mtime and GGUF header.
///
/// A GGUF header that can't be parsed is not an error: the file is still
/// usable, its metadata just stays unknown.
fn read_file_info(
    path: &Path,
    size: u64,
    modified: Option<SystemTime>,
) -> Result<FileInfo, String> {
    let mut file = File::open(path).map_err(read_error)?;
    let mut magic = [0u8; 4];
    let is_gguf = match file.read_exact(&mut magic) {
        Ok(()) => &magic == b"GGUF",
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => false,
        Err(error) => return Err(read_error(error)),
    };
    let meta = if is_gguf {
        read_gguf_meta(path).ok()
    } else {
        None
    };
    Ok(FileInfo {
        size,
        modified,
        meta,
    })
}

/// Which model slot a row belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotId {
    /// The main text model.
    Text,
    /// The vision model.
    Vision,
    /// The vision model's projector.
    VisionProjector,
    /// The embedding model.
    Embedding,
    /// The speech-to-text model.
    SpeechToText,
    /// The text-to-speech voice.
    TextToSpeech,
}

/// One model's line in the memory card.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ModelRow {
    /// The slot the file is configured for.
    pub slot: SlotId,
    /// The file name shown to the user.
    pub file_name: String,
    /// The weights: the file size on disk.
    pub weights: u64,
    /// The KV cache, in bytes; 0 when it is unknown or not applicable.
    pub context: u64,
    /// The non-weight overhead, in bytes.
    pub extra: u64,
    /// The part that stays in RAM, in bytes.
    pub ram: u64,
    /// The part that goes to VRAM, in bytes.
    pub vram: u64,
    /// Whether the KV cache size couldn't be computed.
    pub kv_unknown: bool,
    /// How many layers the model has, when known.
    pub layer_count: Option<u64>,
    /// Why the file couldn't be inspected, when it couldn't.
    pub error: Option<String>,
}

/// The context ceiling and the value the user's request was clamped to.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ContextReport {
    /// Everything known about the context ceiling.
    pub limits: ContextLimits,
    /// What the user asked for, in tokens.
    pub requested: u32,
    /// What will actually be loaded, and why it differs.
    pub applied: ClampedContext,
}

/// The machine as the estimate saw it.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct HardwareSummary {
    /// The processor.
    pub cpu: CpuInfo,
    /// Installed and currently free memory.
    pub ram: RamInfo,
    /// Every detected graphics adapter.
    pub gpus: Vec<GpuInfo>,
}

/// The whole memory forecast.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Estimate {
    /// One row per configured model, in slot order.
    pub rows: Vec<ModelRow>,
    /// Everything every estimate pays for besides the models, in bytes.
    pub fixed_extras: u64,
    /// Total RAM use, in bytes.
    pub ram_used: u64,
    /// Total VRAM use, in bytes.
    pub vram_used: u64,
    /// What the machine may use.
    pub budget: MemoryBudget,
    /// How comfortably the forecast fits.
    pub status: FitStatus,
    /// The short, user-facing fit label.
    pub status_label: String,
    /// The text model's context report, when it is configured and readable.
    pub context: Option<ContextReport>,
    /// Where compaction kicks in, when a context will be loaded.
    pub compaction: Option<CompactionThresholds>,
    /// The machine the forecast was computed against.
    pub hardware: HardwareSummary,
}

/// Builds the full memory forecast.
pub fn compute_estimate(
    settings: &Settings,
    probe: &dyn HardwareProbe,
    inspector: &dyn ModelInspector,
) -> Estimate {
    let ram = probe.ram();
    let gpus = probe.gpus();
    let overrides = BudgetOverrides {
        ram_gb: settings.memory.ram_gb,
        vram_gb: settings.memory.vram_gb,
    };
    let budget = compute_budget(&ram, &gpus, &overrides);

    let models = &settings.models;
    let paths: [(SlotId, Option<&PathBuf>); 6] = [
        (SlotId::Text, models.text.path.as_ref()),
        (SlotId::Vision, models.vision.model_path.as_ref()),
        (
            SlotId::VisionProjector,
            models.vision.projector_path.as_ref(),
        ),
        (SlotId::Embedding, models.embedding.path.as_ref()),
        (SlotId::SpeechToText, models.stt.path.as_ref()),
        (SlotId::TextToSpeech, models.tts.voice_path.as_ref()),
    ];
    let mut targets: Vec<Target<'_>> = Vec::new();
    for (slot, path) in paths {
        if let Some(path) = path {
            targets.push(Target {
                slot,
                path: path.as_path(),
                result: inspector.inspect(path.as_path()),
            });
        }
    }

    // The projector borrows its vision model's layer count.
    let vision_layers = targets
        .iter()
        .find(|target| target.slot == SlotId::Vision)
        .and_then(|target| target.result.as_ref().ok())
        .and_then(|info| info.meta.as_ref())
        .and_then(|meta| meta.block_count);

    // Every row except the text model: its footprint only needs its own
    // settings, so it can be computed before the context sizing.
    let mut rows: Vec<ModelRow> = Vec::new();
    let mut others_ram = 0u64;
    for target in &targets {
        if target.slot == SlotId::Text {
            continue;
        }
        let row = match &target.result {
            Err(error) => error_row(target.slot, target.path, error),
            Ok(info) => {
                let (kind, n_ctx, gpu_layers, split_layers) = match target.slot {
                    SlotId::Vision => (ModelKind::Vision, None, models.vision.gpu_layers, None),
                    SlotId::VisionProjector => (
                        ModelKind::VisionProjector,
                        None,
                        models.vision.gpu_layers,
                        vision_layers,
                    ),
                    SlotId::Embedding => (
                        ModelKind::Embedding,
                        Some(models.embedding.context_size),
                        models.embedding.gpu_layers,
                        None,
                    ),
                    SlotId::SpeechToText => (ModelKind::SpeechToText, None, 0, None),
                    SlotId::TextToSpeech => (ModelKind::TextToSpeech, None, 0, None),
                    SlotId::Text => unreachable!("text rows are handled separately"),
                };
                row_for(
                    target.slot,
                    target.path,
                    info,
                    kind,
                    n_ctx,
                    gpu_layers,
                    split_layers,
                )
            }
        };
        others_ram = others_ram.saturating_add(row.ram);
        if budget.unified {
            others_ram = others_ram.saturating_add(row.vram);
        }
        rows.push(row);
    }

    // The text model, sized against what the other models leave over.
    let mut context = None;
    let mut compaction = None;
    let mut text_row = None;
    if let Some(target) = targets.iter().find(|target| target.slot == SlotId::Text) {
        match &target.result {
            Err(error) => {
                text_row = Some(error_row(SlotId::Text, target.path, error));
            }
            Ok(info) => {
                let prelim =
                    footprint_of(ModelKind::Text, info, None, models.text.gpu_layers, None);
                let usable = usable_memory(budget.ram, others_ram, prelim.weights, prelim.extra);
                let meta = info.meta.as_ref();
                let system_max = meta.and_then(|meta| {
                    system_max_context(meta, KvCacheType::F16, usable, ENGINE_MAX_CONTEXT)
                });
                let model_max = meta.and_then(|meta| meta.context_length);
                let limits = context_limits(model_max, system_max);
                let applied = clamp_requested(models.text.context_size, &limits);
                let row = row_for(
                    SlotId::Text,
                    target.path,
                    info,
                    ModelKind::Text,
                    applied.value,
                    models.text.gpu_layers,
                    None,
                );
                compaction = applied
                    .value
                    .map(|n_ctx| compaction_thresholds(n_ctx, &settings.compaction));
                context = Some(ContextReport {
                    limits,
                    requested: models.text.context_size,
                    applied,
                });
                text_row = Some(row);
            }
        }
    }
    if let Some(row) = text_row {
        rows.insert(0, row);
    }

    let mut ram_sum = 0u64;
    let mut vram_sum = 0u64;
    for row in &rows {
        ram_sum = ram_sum.saturating_add(row.ram);
        vram_sum = vram_sum.saturating_add(row.vram);
    }
    let ram_used = ram_sum
        .saturating_add(FIXED_EXTRAS_BYTES)
        .saturating_add(if budget.unified { vram_sum } else { 0 });
    let vram_used = vram_sum;

    let mut status = combined_status(ram_used, budget.ram, vram_used, budget.vram);
    let no_room = context
        .as_ref()
        .is_some_and(|context| context.applied.reason == ClampReason::NoRoom);
    if no_room {
        status = FitStatus::TooBig;
    }

    Estimate {
        rows,
        fixed_extras: FIXED_EXTRAS_BYTES,
        ram_used,
        vram_used,
        budget,
        status,
        status_label: status.label().to_string(),
        context,
        compaction,
        hardware: HardwareSummary {
            cpu: probe.cpu(),
            ram,
            gpus: gpus.gpus,
        },
    }
}

/// One configured slot and what the inspector said about it.
struct Target<'a> {
    slot: SlotId,
    path: &'a Path,
    result: Result<FileInfo, String>,
}

/// A row whose file couldn't be read: an error and no bytes at all.
fn error_row(slot: SlotId, path: &Path, error: &str) -> ModelRow {
    ModelRow {
        slot,
        file_name: file_name(path),
        weights: 0,
        context: 0,
        extra: 0,
        ram: 0,
        vram: 0,
        kv_unknown: false,
        layer_count: None,
        error: Some(error.to_string()),
    }
}

/// A row from a model's footprint.
fn row_for(
    slot: SlotId,
    path: &Path,
    info: &FileInfo,
    kind: ModelKind,
    n_ctx: Option<u32>,
    gpu_layers: u32,
    split_layers: Option<u64>,
) -> ModelRow {
    let footprint = footprint_of(kind, info, n_ctx, gpu_layers, split_layers);
    ModelRow {
        slot,
        file_name: file_name(path),
        weights: footprint.weights,
        context: footprint.kv.bytes(),
        extra: footprint.extra,
        ram: footprint.ram,
        vram: footprint.vram,
        kv_unknown: matches!(footprint.kv, KvEstimate::Unknown),
        layer_count: footprint.total_layers,
        error: None,
    }
}

/// The forecast for one file, always with a 16-bit KV cache.
fn footprint_of(
    kind: ModelKind,
    info: &FileInfo,
    n_ctx: Option<u32>,
    gpu_layers: u32,
    split_layers: Option<u64>,
) -> Footprint {
    let input = ModelInput {
        kind,
        file_size: info.size,
        meta: info.meta.clone(),
        n_ctx,
        gpu_layers,
        split_layers,
    };
    model_footprint(&input, KvCacheType::F16)
}

/// The name shown for a model file.
fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// The worker's shared state: the newest request, the latest generation and the
/// shutdown flag.
struct WorkerState {
    pending: Option<Job>,
    generation: u64,
    shutdown: bool,
}

/// One request waiting to be computed.
struct Job {
    generation: u64,
    settings: Settings,
}

/// The lock and wake-up signal shared by the estimator and its worker.
struct Shared {
    state: Mutex<WorkerState>,
    cvar: Condvar,
}

impl Shared {
    fn lock_state(&self) -> MutexGuard<'_, WorkerState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Runs estimates on a worker thread, coalescing rapid requests.
///
/// Only the newest pending request is computed: dragging a slider can enqueue
/// any number of requests, and every result older than the latest request is
/// thrown away.
pub struct Estimator {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Estimator {
    /// Starts the worker and returns the handle used to send it requests.
    pub fn spawn(
        probe: Arc<dyn HardwareProbe>,
        inspector: Arc<dyn ModelInspector>,
        sink: Box<dyn Fn(u64, Estimate) + Send + 'static>,
    ) -> Estimator {
        let shared = Arc::new(Shared {
            state: Mutex::new(WorkerState {
                pending: None,
                generation: 0,
                shutdown: false,
            }),
            cvar: Condvar::new(),
        });

        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::spawn(move || {
            run_worker(worker_shared, probe, inspector, sink);
        });

        Estimator {
            shared,
            worker: Some(worker),
        }
    }

    /// Queues `settings`, replacing any request that hasn't started yet.
    ///
    /// Returns the new generation; the first request is generation 1.
    pub fn request(&self, settings: Settings) -> u64 {
        let mut state = self.shared.lock_state();
        state.generation += 1;
        let generation = state.generation;
        state.pending = Some(Job {
            generation,
            settings,
        });
        self.shared.cvar.notify_one();
        generation
    }
}

impl Drop for Estimator {
    fn drop(&mut self) {
        {
            let mut state = self.shared.lock_state();
            state.shutdown = true;
            state.pending = None;
            self.shared.cvar.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// The worker loop: take the newest job, compute it, report it only if it is
/// still the newest request.
fn run_worker(
    shared: Arc<Shared>,
    probe: Arc<dyn HardwareProbe>,
    inspector: Arc<dyn ModelInspector>,
    sink: Box<dyn Fn(u64, Estimate) + Send + 'static>,
) {
    loop {
        let job = {
            let mut state = shared.lock_state();
            loop {
                if state.shutdown {
                    return;
                }
                if let Some(job) = state.pending.take() {
                    break job;
                }
                state = shared
                    .cvar
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        };

        let estimate = compute_estimate(&job.settings, probe.as_ref(), inspector.as_ref());

        let still_current = {
            let state = shared.lock_state();
            !state.shutdown && state.generation == job.generation
        };
        if still_current {
            sink(job.generation, estimate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::testing::FakeProbe;
    use crate::hardware::{GpuKind, GpuReport, UnifiedMemory};
    use crate::settings::{ModelSlots, TextSlot};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    /// Metadata for a 32-layer model with 8 KV heads of 128 dimensions.
    fn meta() -> ModelMeta {
        ModelMeta {
            architecture: Some("llama".to_owned()),
            context_length: Some(131_072),
            block_count: Some(32),
            embedding_length: Some(4096),
            head_count: Some(32),
            head_count_kv: Some(8),
            key_length: Some(128),
            value_length: Some(128),
            sliding_window: None,
        }
    }

    fn file(size: u64, meta: Option<ModelMeta>) -> FileInfo {
        FileInfo {
            size,
            modified: None,
            meta,
        }
    }

    /// An inspector that answers from a fixed map.
    #[derive(Clone, Default)]
    struct FakeInspector {
        files: HashMap<PathBuf, Result<FileInfo, String>>,
    }

    impl FakeInspector {
        fn with(mut self, path: &str, result: Result<FileInfo, String>) -> Self {
            self.files.insert(PathBuf::from(path), result);
            self
        }
    }

    impl ModelInspector for FakeInspector {
        fn inspect(&self, path: &Path) -> Result<FileInfo, String> {
            self.files
                .get(path)
                .cloned()
                .unwrap_or_else(|| Err("File not found".to_string()))
        }
    }

    fn probe(available: u64, gpus: GpuReport) -> FakeProbe {
        FakeProbe::new(
            RamInfo {
                total: 32 * GIB,
                available,
            },
            CpuInfo {
                name: "Test CPU".to_string(),
                logical_threads: 8,
                physical_cores: Some(4),
            },
            gpus,
        )
    }

    fn text_settings(path: &str, context_size: u32, gpu_layers: u32) -> Settings {
        Settings {
            models: ModelSlots {
                text: TextSlot {
                    path: Some(PathBuf::from(path)),
                    gpu_layers,
                    context_size,
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn no_models_means_no_rows_and_only_fixed_extras() {
        let probe = probe(16 * GIB, GpuReport::default());
        let estimate = compute_estimate(&Settings::default(), &probe, &FakeInspector::default());

        assert!(estimate.rows.is_empty());
        assert_eq!(estimate.context, None);
        assert_eq!(estimate.compaction, None);
        assert_eq!(estimate.fixed_extras, FIXED_EXTRAS_BYTES);
        assert_eq!(estimate.ram_used, FIXED_EXTRAS_BYTES);
        assert_eq!(estimate.vram_used, 0);
        assert_eq!(estimate.status, FitStatus::Ok);
        assert_eq!(estimate.status_label, "Fits comfortably");
    }

    #[test]
    fn a_text_model_reports_context_and_compaction() {
        let inspector =
            FakeInspector::default().with("C:/models/text.gguf", Ok(file(GIB, Some(meta()))));
        let settings = text_settings("C:/models/text.gguf", 5000, 0);
        let probe = probe(16 * GIB, GpuReport::default());

        let estimate = compute_estimate(&settings, &probe, &inspector);

        assert_eq!(estimate.rows.len(), 1);
        let row = &estimate.rows[0];
        assert_eq!(row.slot, SlotId::Text);
        assert_eq!(row.file_name, "text.gguf");
        assert!(row.context > 0);
        assert!(!row.kv_unknown);
        assert_eq!(row.layer_count, Some(32));

        let context = estimate.context.expect("a context report");
        assert_eq!(context.requested, 5000);
        assert_eq!(context.applied.value, Some(4864));
        assert_eq!(context.applied.reason, ClampReason::Rounded);
        assert_eq!(
            estimate.compaction,
            Some(compaction_thresholds(4864, &settings.compaction))
        );
        assert_eq!(estimate.compaction.expect("compaction").trigger, 3379);
    }

    #[test]
    fn an_unreadable_file_gets_an_error_row() {
        let inspector = FakeInspector::default().with(
            "C:/models/text.gguf",
            Err("Couldn't read this file".to_string()),
        );
        let settings = text_settings("C:/models/text.gguf", 4096, 0);
        let probe = probe(16 * GIB, GpuReport::default());

        let estimate = compute_estimate(&settings, &probe, &inspector);

        assert_eq!(estimate.rows.len(), 1);
        let row = &estimate.rows[0];
        assert_eq!(row.error.as_deref(), Some("Couldn't read this file"));
        assert_eq!(row.weights, 0);
        assert_eq!(row.context, 0);
        assert_eq!(row.extra, 0);
        assert_eq!(row.ram, 0);
        assert_eq!(row.vram, 0);
        assert_eq!(estimate.context, None);
        assert_eq!(estimate.compaction, None);
    }

    #[test]
    fn unified_memory_counts_vram_against_ram() {
        let gpus = GpuReport {
            gpus: vec![GpuInfo {
                name: "Apple GPU".to_string(),
                vendor: Some("Apple".to_string()),
                kind: GpuKind::Integrated,
                dedicated_vram: None,
            }],
            unified_memory: Some(UnifiedMemory {
                gpu_working_set_limit: Some(10 * GIB),
            }),
        };
        let inspector =
            FakeInspector::default().with("C:/models/text.gguf", Ok(file(4 * GIB, Some(meta()))));
        let settings = text_settings("C:/models/text.gguf", 4096, 32);
        let probe = probe(32 * GIB, gpus);

        let estimate = compute_estimate(&settings, &probe, &inspector);

        assert!(estimate.budget.unified);
        let row = &estimate.rows[0];
        assert!(row.vram > 0, "all layers should be offloaded");
        assert_eq!(estimate.vram_used, row.vram);
        assert_eq!(estimate.ram_used, row.ram + FIXED_EXTRAS_BYTES + row.vram);
    }

    #[test]
    fn no_room_for_a_context_is_too_big() {
        let inspector =
            FakeInspector::default().with("C:/models/text.gguf", Ok(file(4 * GIB, Some(meta()))));
        let settings = text_settings("C:/models/text.gguf", 4096, 0);
        let probe = probe(MIB, GpuReport::default());

        let estimate = compute_estimate(&settings, &probe, &inspector);

        let context = estimate.context.expect("a context report");
        assert_eq!(context.applied.reason, ClampReason::NoRoom);
        assert_eq!(context.applied.value, None);
        assert_eq!(context.limits.system_max, Some(0));
        assert_eq!(estimate.compaction, None);
        assert_eq!(estimate.status, FitStatus::TooBig);
    }

    #[test]
    fn ram_is_read_on_every_compute() {
        let inspector =
            FakeInspector::default().with("C:/models/text.gguf", Ok(file(GIB, Some(meta()))));
        let settings = text_settings("C:/models/text.gguf", 4096, 0);
        let probe = probe(16 * GIB, GpuReport::default());

        let _ = compute_estimate(&settings, &probe, &inspector);
        let _ = compute_estimate(&settings, &probe, &inspector);

        assert_eq!(probe.ram_calls(), 2);
    }

    /// An inspector whose first call blocks until the test releases it.
    struct BlockingInspector {
        started: Mutex<Option<mpsc::Sender<()>>>,
        release: Mutex<mpsc::Receiver<()>>,
        calls: AtomicU32,
        info: FileInfo,
    }

    impl ModelInspector for BlockingInspector {
        fn inspect(&self, _path: &Path) -> Result<FileInfo, String> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                if let Some(started) = self
                    .started
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                {
                    let _ = started.send(());
                }
                let _ = self
                    .release
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .recv();
            }
            Ok(self.info.clone())
        }
    }

    #[test]
    fn only_the_newest_request_is_computed() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let inspector = Arc::new(BlockingInspector {
            started: Mutex::new(Some(started_tx)),
            release: Mutex::new(release_rx),
            calls: AtomicU32::new(0),
            info: file(GIB, Some(meta())),
        });
        let probe = Arc::new(probe(16 * GIB, GpuReport::default()));

        let (sink_tx, sink_rx) = mpsc::channel();
        let estimator = Estimator::spawn(
            probe,
            inspector.clone(),
            Box::new(move |generation, estimate| {
                let _ = sink_tx.send((generation, estimate));
            }),
        );

        let settings = text_settings("C:/models/text.gguf", 4096, 0);
        assert_eq!(estimator.request(settings.clone()), 1);

        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the worker should start computing");

        for _ in 0..9 {
            estimator.request(settings.clone());
        }

        release_tx.send(()).expect("release the inspector");

        let (generation, _estimate) = sink_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("exactly one result should arrive");
        assert_eq!(generation, 10);

        assert!(
            sink_rx.recv_timeout(Duration::from_secs(1)).is_err(),
            "a stale result must never reach the sink"
        );
        assert!(
            inspector.calls.load(Ordering::SeqCst) <= 2,
            "at most two computations should run"
        );

        drop(estimator);
    }

    /// A tiny GGUF file with only `general.architecture`.
    fn gguf_bytes(architecture: &str) -> Vec<u8> {
        let key = "general.architecture";
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
        bytes.extend_from_slice(architecture.as_bytes());
        bytes
    }

    #[test]
    fn fs_inspector_reads_gguf_metadata() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model.gguf");
        let bytes = gguf_bytes("llama");
        std::fs::write(&path, &bytes).expect("write fixture");

        let inspector = FsInspector::new();
        let info = inspector.inspect(&path).expect("the file is readable");

        assert_eq!(info.size, bytes.len() as u64);
        assert!(info.modified.is_some());
        let meta = info.meta.expect("GGUF metadata");
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn fs_inspector_leaves_non_gguf_metadata_unknown() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("voice.onnx");
        let bytes = b"this is not a model";
        std::fs::write(&path, bytes).expect("write fixture");

        let info = FsInspector::new()
            .inspect(&path)
            .expect("the file is readable");

        assert_eq!(info.size, bytes.len() as u64);
        assert_eq!(info.meta, None);
    }

    #[test]
    fn fs_inspector_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let error = FsInspector::new()
            .inspect(&dir.path().join("missing.gguf"))
            .expect_err("the file does not exist");

        assert_eq!(error, "File not found");
    }
}
