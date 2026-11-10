//! Forecasts how much memory a model needs before it is loaded.
//!
//! Everything here is integer arithmetic: sizes are `u64` bytes and every
//! intermediate product is widened to `u128`, so a huge file or context can't
//! wrap. Division that could lose a byte rounds *up*: a forecast that is one
//! byte too low is worse than one that is one byte too high.
//!
//! Missing metadata means "unknown", never "free". A model whose header
//! doesn't say how big its KV cache is gets [`KvEstimate::Unknown`], and the
//! number it contributes to the total is 0 because there is nothing better to
//! say; the estimator decides what to do about that.

use crate::gguf::ModelMeta;

/// One mebibyte, in bytes.
pub const MIB: u64 = 1 << 20;
/// One gibibyte, in bytes.
pub const GIB: u64 = 1 << 30;

/// The app and its WebView, in bytes (an estimate).
pub const APP_BASE_BYTES: u64 = 300 * MIB;
/// The audio capture and playback buffers, in bytes (an estimate).
pub const AUDIO_BUFFERS_BYTES: u64 = 16 * MIB;
/// Everything every estimate pays for besides the models themselves.
pub const FIXED_EXTRAS_BYTES: u64 = APP_BASE_BYTES + AUDIO_BUFFERS_BYTES;

/// How much memory each KV cache element costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheType {
    /// A 16-bit cache: two bytes per element.
    F16,
    /// An 8-bit cache in 32-element blocks: 34 bytes per block.
    Q8_0,
    /// A 4-bit cache in 32-element blocks: 18 bytes per block.
    Q4_0,
}

impl KvCacheType {
    /// The bytes per element, as an exact fraction `(numerator, denominator)`.
    pub fn bytes_per_element(self) -> (u64, u64) {
        match self {
            KvCacheType::F16 => (2, 1),
            KvCacheType::Q8_0 => (34, 32),
            KvCacheType::Q4_0 => (18, 32),
        }
    }
}

/// The KV cache size for `n_ctx` tokens, in bytes, rounded up.
///
/// Returns `None` when any of the inputs is unknown or zero: a cache whose
/// size can't be computed is unknown, not free.
pub fn kv_cache_bytes(meta: &ModelMeta, n_ctx: u32, kv: KvCacheType) -> Option<u64> {
    let layers = meta.block_count.filter(|layers| *layers > 0)? as u128;
    let heads = meta.kv_heads().filter(|heads| *heads > 0)? as u128;
    let dims = meta.key_dim()?.checked_add(meta.value_dim()?)?;
    if n_ctx == 0 || dims == 0 {
        return None;
    }
    let (numerator, denominator) = kv.bytes_per_element();
    let elements = layers
        .checked_mul(n_ctx as u128)?
        .checked_mul(heads)?
        .checked_mul(dims as u128)?;
    let bytes = ceil_div(
        elements.checked_mul(numerator as u128)?,
        denominator as u128,
    );
    u64::try_from(bytes).ok()
}

/// The non-weight memory a loaded model needs, in bytes.
///
/// That is the larger of 256 MiB and 5% of the weights, rounded up. It stays
/// in RAM even when every layer is offloaded.
pub fn overhead_bytes(weights: u64) -> u64 {
    let ratio = u128::from(weights) * 5;
    let five_percent = ceil_div(ratio, 100);
    let floor = u128::from(256 * MIB);
    u64::try_from(five_percent.max(floor)).unwrap_or(u64::MAX)
}

/// What kind of file a footprint is being computed for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// The main text model.
    Text,
    /// An embedding model for long-term memory.
    Embedding,
    /// A vision model.
    Vision,
    /// A companion file for the vision model, such as its projector.
    VisionProjector,
    /// A speech-to-text model.
    SpeechToText,
    /// A text-to-speech model.
    TextToSpeech,
}

impl ModelKind {
    /// Whether this kind's file carries usable GGUF metadata at all.
    fn uses_metadata(self) -> bool {
        !matches!(self, ModelKind::SpeechToText | ModelKind::TextToSpeech)
    }

    /// Whether this kind has a KV cache of its own.
    fn has_kv_cache(self) -> bool {
        matches!(self, ModelKind::Text | ModelKind::Embedding)
    }
}

/// One model file, as far as the forecast is concerned.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ModelInput {
    /// What the file is for.
    pub kind: ModelKind,
    /// Its size on disk, in bytes.
    pub file_size: u64,
    /// Its GGUF metadata, when the header could be read.
    pub meta: Option<ModelMeta>,
    /// The context it will be loaded with, in tokens.
    pub n_ctx: Option<u32>,
    /// How many layers to offload to the GPU.
    pub gpu_layers: u32,
    /// The layer count to divide the offload by, when it isn't the file's own.
    ///
    /// A vision projector has no layers of its own; this lets it borrow its
    /// vision model's.
    pub split_layers: Option<u64>,
}

impl ModelInput {
    /// The metadata that can be trusted, if any.
    fn usable_meta(&self) -> Option<&ModelMeta> {
        if !self.kind.uses_metadata() {
            return None;
        }
        self.meta
            .as_ref()
            .filter(|meta| meta.block_count.is_some_and(|layers| layers > 0))
    }
}

/// What is known about a model's KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvEstimate {
    /// This kind of file has no KV cache of its own.
    NotApplicable,
    /// The cache size, in bytes.
    Known(u64),
    /// It has a cache, but its size couldn't be computed.
    Unknown,
}

impl KvEstimate {
    /// The bytes this estimate contributes to the total.
    pub fn bytes(self) -> u64 {
        match self {
            KvEstimate::Known(bytes) => bytes,
            KvEstimate::NotApplicable | KvEstimate::Unknown => 0,
        }
    }
}

/// A model's forecast memory use, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Footprint {
    /// The weights: the file size on disk.
    pub weights: u64,
    /// The KV cache.
    pub kv: KvEstimate,
    /// The non-weight overhead, always in RAM.
    pub extra: u64,
    /// The part that stays in RAM.
    pub ram: u64,
    /// The part that goes to VRAM.
    pub vram: u64,
    /// How many layers the GPU takes.
    pub offloaded_layers: u64,
    /// How many layers there are to divide, when known.
    pub total_layers: Option<u64>,
}

/// Forecasts one model's memory use, splitting it between RAM and VRAM.
pub fn model_footprint(input: &ModelInput, kv: KvCacheType) -> Footprint {
    let weights = input.file_size;
    let (kv_estimate, extra) = match input.usable_meta() {
        None => {
            // Without usable metadata, assume a fifth of the file on top: the
            // total is then 1.2 × the file size, with no KV of its own.
            (
                KvEstimate::NotApplicable,
                ceil_div(u128::from(weights), 5) as u64,
            )
        }
        Some(meta) => {
            let estimate = if input.kind.has_kv_cache() {
                match input.n_ctx {
                    Some(n_ctx) => match kv_cache_bytes(meta, n_ctx, kv) {
                        Some(bytes) => KvEstimate::Known(bytes),
                        None => KvEstimate::Unknown,
                    },
                    None => KvEstimate::Unknown,
                }
            } else {
                KvEstimate::NotApplicable
            };
            (estimate, overhead_bytes(weights))
        }
    };

    let total_layers = input
        .split_layers
        .or_else(|| input.usable_meta().and_then(|meta| meta.block_count));
    let offloaded = total_layers
        .filter(|layers| *layers > 0)
        .map_or(0, |layers| u64::from(input.gpu_layers).min(layers));

    let split = u128::from(weights) + u128::from(kv_estimate.bytes());
    let (ram, vram) = match total_layers.filter(|layers| *layers > 0) {
        Some(layers) if offloaded > 0 => {
            let vram = ceil_div(split * u128::from(offloaded), u128::from(layers));
            let vram = u64::try_from(vram).unwrap_or(u64::MAX);
            (weights + kv_estimate.bytes() - vram + extra, vram)
        }
        _ => (weights + kv_estimate.bytes() + extra, 0),
    };

    Footprint {
        weights,
        kv: kv_estimate,
        extra,
        ram,
        vram,
        offloaded_layers: offloaded,
        total_layers,
    }
}

/// How comfortably something fits in one memory pool.
///
/// The order is the severity order, so the worse of two statuses is `max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitStatus {
    /// Under 70% of the budget.
    Ok,
    /// 70% to 90% of the budget.
    Tight,
    /// More than the machine has.
    TooBig,
}

impl FitStatus {
    /// The short label shown to the user.
    pub fn label(self) -> &'static str {
        match self {
            FitStatus::Ok => "Fits comfortably",
            FitStatus::Tight => "Tight fit",
            FitStatus::TooBig => "Too big for this PC",
        }
    }
}

/// The fit status of one pool.
///
/// Using nothing always fits, even with no budget at all; needing anything at
/// all from a pool of zero doesn't.
pub fn pool_status(used: u64, budget: u64) -> FitStatus {
    if used == 0 {
        return FitStatus::Ok;
    }
    if budget == 0 {
        return FitStatus::TooBig;
    }
    let used = u128::from(used) * 100;
    let budget = u128::from(budget);
    if used < budget * 70 {
        FitStatus::Ok
    } else if used <= budget * 90 {
        FitStatus::Tight
    } else {
        FitStatus::TooBig
    }
}

/// The worse of the RAM and VRAM fit statuses.
pub fn combined_status(
    ram_used: u64,
    ram_budget: u64,
    vram_used: u64,
    vram_budget: u64,
) -> FitStatus {
    pool_status(ram_used, ram_budget).max(pool_status(vram_used, vram_budget))
}

/// `numerator ÷ denominator`, rounded up.
///
/// The caller must not pass a zero denominator.
fn ceil_div(numerator: u128, denominator: u128) -> u128 {
    numerator.div_ceil(denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn text_input(file_size: u64, gpu_layers: u32) -> ModelInput {
        ModelInput {
            kind: ModelKind::Text,
            file_size,
            meta: Some(meta()),
            n_ctx: Some(4096),
            gpu_layers,
            split_layers: None,
        }
    }

    #[test]
    fn kv_cache_sizes_match_the_spec() {
        assert_eq!(
            kv_cache_bytes(&meta(), 4096, KvCacheType::F16),
            Some(536_870_912)
        );
        assert_eq!(
            kv_cache_bytes(&meta(), 4096, KvCacheType::Q8_0),
            Some(285_212_672)
        );
        assert_eq!(
            kv_cache_bytes(&meta(), 4096, KvCacheType::Q4_0),
            Some(150_994_944)
        );
    }

    #[test]
    fn kv_cache_rounds_up() {
        let meta = ModelMeta {
            block_count: Some(1),
            head_count_kv: Some(1),
            key_length: Some(1),
            value_length: Some(0),
            ..ModelMeta::default()
        };
        assert_eq!(kv_cache_bytes(&meta, 1, KvCacheType::Q8_0), Some(2));
    }

    #[test]
    fn kv_cache_is_unknown_without_head_counts() {
        let meta = ModelMeta {
            block_count: Some(32),
            ..ModelMeta::default()
        };
        assert_eq!(kv_cache_bytes(&meta, 4096, KvCacheType::F16), None);
        assert_eq!(kv_cache_bytes(&meta, 0, KvCacheType::F16), None);
    }

    #[test]
    fn overhead_has_a_floor_and_a_ratio() {
        assert_eq!(overhead_bytes(GIB), 268_435_456);
        assert_eq!(overhead_bytes(10 * GIB), 536_870_912);
        assert_eq!(overhead_bytes(0), 268_435_456);
    }

    #[test]
    fn models_without_metadata_cost_one_point_two_times_the_file() {
        let input = ModelInput {
            kind: ModelKind::TextToSpeech,
            file_size: 100_000_000,
            meta: None,
            n_ctx: None,
            gpu_layers: 99,
            split_layers: None,
        };
        let footprint = model_footprint(&input, KvCacheType::F16);
        assert_eq!(footprint.weights, 100_000_000);
        assert_eq!(footprint.kv, KvEstimate::NotApplicable);
        assert_eq!(footprint.ram, 120_000_000);
        assert_eq!(footprint.vram, 0);
        assert_eq!(footprint.offloaded_layers, 0);
        assert_eq!(footprint.total_layers, None);
    }

    #[test]
    fn gpu_split_moves_a_share_of_weights_and_kv() {
        let half = model_footprint(&text_input(4 * GIB, 16), KvCacheType::F16);
        assert_eq!(half.kv, KvEstimate::Known(536_870_912));
        assert_eq!(half.extra, 268_435_456);
        assert_eq!(half.offloaded_layers, 16);
        assert_eq!(half.total_layers, Some(32));
        assert_eq!(half.vram, 2_415_919_104);
        assert_eq!(half.ram, 2_684_354_560);

        let all = model_footprint(&text_input(4 * GIB, 99), KvCacheType::F16);
        assert_eq!(all.offloaded_layers, 32);
        assert_eq!(all.vram, 4_831_838_208);
        assert_eq!(all.ram, 268_435_456);
    }

    #[test]
    fn split_layers_lets_a_projector_borrow_its_models_layers() {
        let input = ModelInput {
            kind: ModelKind::VisionProjector,
            file_size: 100 * MIB,
            meta: Some(ModelMeta {
                block_count: Some(32),
                ..ModelMeta::default()
            }),
            n_ctx: None,
            gpu_layers: 32,
            split_layers: Some(32),
        };
        let footprint = model_footprint(&input, KvCacheType::F16);
        assert_eq!(footprint.kv, KvEstimate::NotApplicable);
        assert_eq!(footprint.extra, 256 * MIB);
        assert_eq!(footprint.offloaded_layers, 32);
        assert_eq!(footprint.vram, 100 * MIB);
        assert_eq!(footprint.ram, 256 * MIB);
    }

    #[test]
    fn unknown_kv_counts_as_zero_bytes() {
        let input = ModelInput {
            kind: ModelKind::Text,
            file_size: GIB,
            meta: Some(ModelMeta {
                block_count: Some(32),
                ..ModelMeta::default()
            }),
            n_ctx: Some(4096),
            gpu_layers: 0,
            split_layers: None,
        };
        let footprint = model_footprint(&input, KvCacheType::F16);
        assert_eq!(footprint.kv, KvEstimate::Unknown);
        assert_eq!(footprint.ram, GIB + 268_435_456);
    }

    #[test]
    fn missing_context_leaves_kv_unknown() {
        let mut input = text_input(GIB, 0);
        input.n_ctx = None;
        let footprint = model_footprint(&input, KvCacheType::F16);
        assert_eq!(footprint.kv, KvEstimate::Unknown);
        assert_eq!(footprint.extra, 268_435_456);
    }

    #[test]
    fn pool_status_boundaries() {
        assert_eq!(pool_status(69, 100), FitStatus::Ok);
        assert_eq!(pool_status(70, 100), FitStatus::Tight);
        assert_eq!(pool_status(90, 100), FitStatus::Tight);
        assert_eq!(pool_status(91, 100), FitStatus::TooBig);
        assert_eq!(pool_status(1, 0), FitStatus::TooBig);
        assert_eq!(pool_status(0, 0), FitStatus::Ok);
        assert_eq!(pool_status(0, 100), FitStatus::Ok);
    }

    #[test]
    fn combined_status_takes_the_worse_pool() {
        assert_eq!(combined_status(10, 100, 95, 100), FitStatus::TooBig);
        assert_eq!(combined_status(75, 100, 10, 100), FitStatus::Tight);
        assert_eq!(combined_status(10, 100, 20, 100), FitStatus::Ok);
        assert_eq!(combined_status(10, 100, 1, 0), FitStatus::TooBig);
    }

    #[test]
    fn labels_are_user_facing() {
        assert_eq!(FitStatus::Ok.label(), "Fits comfortably");
        assert_eq!(FitStatus::Tight.label(), "Tight fit");
        assert_eq!(FitStatus::TooBig.label(), "Too big for this PC");
        assert!(FitStatus::Ok < FitStatus::Tight);
        assert!(FitStatus::Tight < FitStatus::TooBig);
    }
}
