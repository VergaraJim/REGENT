//! Pure voice-input logic (spec §5.8): no I/O, no threads, no audio hardware.
//!
//! Later tasks build the voice service on top of these pieces:
//!
//! - [`resample`]: downmixing a device buffer to mono and a streaming
//!   resampler that converts it to the rate STT expects without clicks at
//!   buffer boundaries.
//! - [`vad`]: an energy-based voice activity detector behind
//!   [`vad::VoiceDetector`], so a model-based detector can replace it later.
//! - [`post_filter`]: dropping the stock phrases STT models emit on silence.
//!
//! Everything here assumes mono `f32` audio at [`STT_SAMPLE_RATE`] once it
//! reaches the detector.

pub mod post_filter;
pub mod resample;
pub mod vad;

pub use post_filter::clean_transcript;
pub use resample::{downmix, Resampler};
pub use vad::{rms, EnergyVad, Utterance, VoiceDetector};

/// The sample rate speech-to-text expects: 16 kHz mono.
pub const STT_SAMPLE_RATE: u32 = 16_000;
