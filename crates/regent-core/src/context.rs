//! Works out how large a context the text model can have on this PC, how the
//! user's requested context is clamped, and where compaction kicks in.
//!
//! The arithmetic mirrors the memory forecast: sizes are integers, products are
//! widened to `u128` so a huge budget or context can't wrap, and divisions that
//! could lose a byte round down only where a smaller context is the safe answer.
//! A context is only ever a multiple of [`CONTEXT_STEP`] except when the user's
//! request is capped by an effective maximum that isn't on the step; that cap is
//! used as-is.

use crate::footprint::{kv_cache_bytes, KvCacheType};
use crate::gguf::ModelMeta;
use crate::settings::CompactionSettings;

/// Context sizes move in steps of this many tokens.
pub const CONTEXT_STEP: u32 = 256;
/// The smallest context that can be loaded.
pub const MIN_CONTEXT: u32 = 256;
/// The largest context we ever ask llama.cpp for.
pub const ENGINE_MAX_CONTEXT: u32 = 1 << 20;
/// Tokens kept free for the reply.
pub const REPLY_RESERVE: u32 = 512;

/// The RAM left for the text model's context, in bytes.
///
/// That is 85% of the RAM budget minus every other model, minus the text
/// model's weights and its non-weight overhead. The 15% held back covers the
/// OS, other apps and allocations the estimate can't see. The result never goes
/// below zero.
pub fn usable_memory(
    ram_budget: u64,
    other_models_ram: u64,
    text_weights: u64,
    text_extra: u64,
) -> u64 {
    let share = u128::from(ram_budget) * 85 / 100;
    let spent = u128::from(other_models_ram) + u128::from(text_weights) + u128::from(text_extra);
    u64::try_from(share.saturating_sub(spent)).unwrap_or(u64::MAX)
}

/// The largest context the KV cache leaves room for, in tokens.
///
/// The answer is the largest multiple of [`CONTEXT_STEP`] up to `engine_max`
/// whose KV cache fits in `usable` bytes. It may be 0: with no room at all the
/// cache still "fits" at zero tokens, and `context_limits` decides whether that
/// is loadable. Returns `None` when the cache size can't be computed at all.
pub fn system_max_context(
    meta: &ModelMeta,
    kv: KvCacheType,
    usable: u64,
    engine_max: u32,
) -> Option<u32> {
    // A cache we can't size at the smallest step can't be sized at any step.
    kv_cache_bytes(meta, MIN_CONTEXT, kv)?;

    // KV grows linearly with the context, so the fitting steps form a prefix
    // of `0..=max_steps`. Step 0 always fits: it needs no cache.
    let max_steps = engine_max / CONTEXT_STEP;
    let mut lo = 0u32;
    let mut hi = max_steps;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if step_fits(meta, kv, mid, usable) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo * CONTEXT_STEP)
}

/// Whether the context at step `steps` fits in `usable` bytes.
fn step_fits(meta: &ModelMeta, kv: KvCacheType, steps: u32, usable: u64) -> bool {
    if steps == 0 {
        return true;
    }
    kv_cache_bytes(meta, steps * CONTEXT_STEP, kv).is_some_and(|bytes| bytes <= usable)
}

/// What holds the context back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitingFactor {
    /// The model was trained for less than this PC can give it.
    Model,
    /// This PC has less room than the model could use.
    ThisPc,
    /// Both are the same.
    Equal,
    /// At least one of them is unknown.
    Unknown,
}

/// Everything known about the context ceiling for the text model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ContextLimits {
    /// The trained context length, if known.
    pub model_max: Option<u32>,
    /// What this PC's memory allows, if it could be computed.
    pub system_max: Option<u32>,
    /// The smaller of the known values.
    pub effective_max: Option<u32>,
    /// Which ceiling is doing the limiting.
    pub limiting: LimitingFactor,
    /// Whether any context at all can be loaded.
    pub fits: bool,
}

/// Combines the model's and this PC's ceilings into one usable limit.
pub fn context_limits(model_max: Option<u64>, system_max: Option<u32>) -> ContextLimits {
    let model_max = model_max.map(|max| u32::try_from(max).unwrap_or(u32::MAX));
    let effective_max = match (model_max, system_max) {
        (Some(model), Some(system)) => Some(model.min(system)),
        (Some(model), None) => Some(model),
        (None, Some(system)) => Some(system),
        (None, None) => None,
    };
    let limiting = match (model_max, system_max) {
        (Some(model), Some(system)) if model < system => LimitingFactor::Model,
        (Some(model), Some(system)) if system < model => LimitingFactor::ThisPc,
        (Some(_), Some(_)) => LimitingFactor::Equal,
        _ => LimitingFactor::Unknown,
    };
    // A zero or tiny context must never be loaded.
    let fits = effective_max.is_none_or(|max| max >= MIN_CONTEXT);

    ContextLimits {
        model_max,
        system_max,
        effective_max,
        limiting,
        fits,
    }
}

/// Why the requested context changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClampReason {
    /// The request was already on the step and within the limits.
    WithinLimits,
    /// It was rounded down to the step.
    Rounded,
    /// It was below the minimum and raised.
    RaisedToMinimum,
    /// It was above the effective maximum and capped.
    AboveMax,
    /// No context can be loaded at all.
    NoRoom,
}

/// The context to load, and why it isn't what the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ClampedContext {
    /// The context to load, or `None` when there is no room.
    pub value: Option<u32>,
    /// Why the value differs from the request.
    pub reason: ClampReason,
}

/// Turns a requested context into one that can actually be loaded.
///
/// The steps run in order, so the last one to change the value names the reason.
pub fn clamp_requested(requested: u32, limits: &ContextLimits) -> ClampedContext {
    if !limits.fits {
        return ClampedContext {
            value: None,
            reason: ClampReason::NoRoom,
        };
    }

    let rounded = requested / CONTEXT_STEP * CONTEXT_STEP;
    let mut reason = if rounded == requested {
        ClampReason::WithinLimits
    } else {
        ClampReason::Rounded
    };
    let mut value = rounded;

    if value < MIN_CONTEXT {
        value = MIN_CONTEXT;
        reason = ClampReason::RaisedToMinimum;
    }

    if let Some(max) = limits.effective_max {
        if value > max {
            // The cap is used as-is, even when it isn't on the 256 step.
            value = max;
            reason = ClampReason::AboveMax;
        }
    }

    ClampedContext {
        value: Some(value),
        reason,
    }
}

/// Where compaction and the reply reserve fall in a context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct CompactionThresholds {
    /// History size that triggers compaction, in tokens.
    pub trigger: u32,
    /// History size kept after compaction, in tokens.
    pub target: u32,
    /// The most history the window can ever hold, in tokens.
    pub hard_limit: u32,
}

/// Works out the compaction thresholds for a context window.
///
/// The percentages come from the settings' validated pair, so an invalid pair
/// behaves exactly like the default. Every subtraction keeps the reply reserve
/// free and saturates at zero.
pub fn compaction_thresholds(n_ctx: u32, compaction: &CompactionSettings) -> CompactionThresholds {
    let (trigger_percent, target_percent) = compaction.validated();
    let n_ctx = u128::from(n_ctx);
    let reserved = u128::from(REPLY_RESERVE);
    let at = |percent: u8| {
        u32::try_from((n_ctx * u128::from(percent) / 100).saturating_sub(reserved))
            .unwrap_or(u32::MAX)
    };

    CompactionThresholds {
        trigger: at(trigger_percent),
        target: at(target_percent),
        hard_limit: u32::try_from(n_ctx.saturating_sub(reserved)).unwrap_or(u32::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metadata for 32 layers, 8 KV heads, key 128 and value 128: 131,072 bytes
    /// per token at F16.
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

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    #[test]
    fn system_max_uses_every_token_of_a_gibibyte() {
        assert_eq!(
            system_max_context(&meta(), KvCacheType::F16, GIB, ENGINE_MAX_CONTEXT),
            Some(8192)
        );
    }

    #[test]
    fn system_max_rounds_down_to_the_step() {
        assert_eq!(
            system_max_context(&meta(), KvCacheType::F16, GIB - 1, ENGINE_MAX_CONTEXT),
            Some(7936)
        );
    }

    #[test]
    fn system_max_can_be_zero() {
        assert_eq!(
            system_max_context(&meta(), KvCacheType::F16, 0, ENGINE_MAX_CONTEXT),
            Some(0)
        );
    }

    #[test]
    fn system_max_respects_the_engine_cap() {
        assert_eq!(
            system_max_context(&meta(), KvCacheType::F16, GIB, 4000),
            Some(3840)
        );
    }

    #[test]
    fn system_max_is_unknown_without_kv_heads() {
        let meta = ModelMeta {
            block_count: Some(32),
            key_length: Some(128),
            value_length: Some(128),
            ..ModelMeta::default()
        };
        assert_eq!(
            system_max_context(&meta, KvCacheType::F16, GIB, ENGINE_MAX_CONTEXT),
            None
        );
    }

    #[test]
    fn usable_memory_holds_back_fifteen_percent() {
        assert_eq!(
            usable_memory(16 * GIB, GIB, 4 * GIB, 256 * MIB),
            8_965_744_230
        );
        assert_eq!(usable_memory(GIB, 0, 4 * GIB, 0), 0);
    }

    #[test]
    fn limits_name_the_limiting_side() {
        let system = context_limits(Some(32768), Some(8192));
        assert_eq!(system.effective_max, Some(8192));
        assert_eq!(system.limiting, LimitingFactor::ThisPc);
        assert!(system.fits);

        let model = context_limits(Some(4096), Some(8192));
        assert_eq!(model.effective_max, Some(4096));
        assert_eq!(model.limiting, LimitingFactor::Model);

        let equal = context_limits(Some(8192), Some(8192));
        assert_eq!(equal.effective_max, Some(8192));
        assert_eq!(equal.limiting, LimitingFactor::Equal);

        let unknown_system = context_limits(None, Some(8192));
        assert_eq!(unknown_system.model_max, None);
        assert_eq!(unknown_system.effective_max, Some(8192));
        assert_eq!(unknown_system.limiting, LimitingFactor::Unknown);
        assert!(unknown_system.fits);

        let nothing = context_limits(None, None);
        assert_eq!(nothing.effective_max, None);
        assert_eq!(nothing.limiting, LimitingFactor::Unknown);
        assert!(nothing.fits);
    }

    #[test]
    fn a_tiny_effective_max_does_not_fit() {
        let limits = context_limits(Some(100), Some(0));
        assert_eq!(limits.effective_max, Some(0));
        assert!(!limits.fits);
    }

    #[test]
    fn clamping_within_a_normal_window() {
        let limits = context_limits(Some(8192), Some(8192));
        let case = |requested| clamp_requested(requested, &limits);

        assert_eq!(
            case(5000),
            ClampedContext {
                value: Some(4864),
                reason: ClampReason::Rounded,
            }
        );
        assert_eq!(
            case(4096),
            ClampedContext {
                value: Some(4096),
                reason: ClampReason::WithinLimits,
            }
        );
        assert_eq!(
            case(100),
            ClampedContext {
                value: Some(MIN_CONTEXT),
                reason: ClampReason::RaisedToMinimum,
            }
        );
        assert_eq!(
            case(9000),
            ClampedContext {
                value: Some(8192),
                reason: ClampReason::AboveMax,
            }
        );
    }

    #[test]
    fn the_cap_is_used_even_off_the_step() {
        let limits = context_limits(Some(3000), Some(8192));
        assert_eq!(
            clamp_requested(4096, &limits),
            ClampedContext {
                value: Some(3000),
                reason: ClampReason::AboveMax,
            }
        );
    }

    #[test]
    fn clamping_reports_no_room() {
        let limits = context_limits(Some(0), None);
        assert_eq!(
            clamp_requested(4096, &limits),
            ClampedContext {
                value: None,
                reason: ClampReason::NoRoom,
            }
        );
    }

    #[test]
    fn an_unknown_ceiling_keeps_the_step() {
        let limits = context_limits(None, None);
        assert_eq!(
            clamp_requested(70000, &limits),
            ClampedContext {
                value: Some(69888),
                reason: ClampReason::Rounded,
            }
        );
    }

    #[test]
    fn thresholds_reserve_the_reply() {
        let thresholds = compaction_thresholds(4096, &CompactionSettings::default());
        assert_eq!(thresholds.trigger, 2764);
        assert_eq!(thresholds.target, 1331);
        assert_eq!(thresholds.hard_limit, 3584);
    }

    #[test]
    fn invalid_percentages_fall_back_to_the_defaults() {
        let invalid = CompactionSettings {
            trigger_percent: 45,
            target_percent: 80,
        };
        assert_eq!(
            compaction_thresholds(4096, &invalid),
            compaction_thresholds(4096, &CompactionSettings::default())
        );
    }

    #[test]
    fn a_tiny_window_saturates_at_zero() {
        let thresholds = compaction_thresholds(256, &CompactionSettings::default());
        assert_eq!(thresholds.trigger, 0);
        assert_eq!(thresholds.target, 0);
        assert_eq!(thresholds.hard_limit, 0);
    }
}
