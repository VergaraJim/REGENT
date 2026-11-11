//! Real audio devices for the voice services (spec §5.8 and §5.9), built on
//! `cpal` behind the `audio` feature.
//!
//! [`mic_factory`] opens the default input device and hands mono 16 kHz samples
//! to the voice service; [`audio_out_factory`] plays mono audio on the default
//! output device. Both are factories because a device may be closed and opened
//! again (turning the microphone off and on, for example).
//!
//! The device code lives in [`capture`] and [`playback`]; this module holds the
//! pieces that are pure enough to test without hardware: sample conversion and
//! the mapping from a raw device error to a message fit for the user.

use std::time::Duration;

pub mod capture;
pub mod playback;

pub use capture::mic_factory;
pub use playback::audio_out_factory;

/// Shown when the default input device cannot be found.
pub const MIC_NONE: &str = "No microphone was found. Plug one in and turn the mic on again.";

/// Shown when the system refuses microphone access.
pub const MIC_DENIED: &str = "Regent isn't allowed to use the microphone. Allow it in your system's privacy settings, then turn the mic on again.";

/// Shown when the microphone exists but cannot be opened.
pub const MIC_OPEN_FAILED: &str =
    "The microphone couldn't be opened. Check that no other app is using it.";

/// Shown when opening the microphone takes too long.
pub const MIC_TIMEOUT: &str = "The microphone didn't respond. Turn the mic on again to retry.";

/// Shown when an open microphone stops working.
pub const MIC_LOST: &str = "The microphone stopped working. Turn the mic on again when it's back.";

/// Shown when the default output device cannot be found.
pub const NO_SPEAKERS: &str = "No speakers or headphones were found.";

/// Shown when this build has no audio support compiled in.
pub const NO_AUDIO: &str = "This build of Regent has no audio support.";

/// How long opening a device may take before it counts as a failure.
pub const DEVICE_OPEN_TIMEOUT: Duration = Duration::from_secs(3);

/// Maps a raw device error to the message the user should see.
///
/// Permission failures are recognised by their text because `cpal` does not
/// report them in a portable way.
pub fn mic_error_message(raw: &str) -> &'static str {
    let raw = raw.to_lowercase();
    if raw.contains("permission") || raw.contains("denied") || raw.contains("not authorized") {
        MIC_DENIED
    } else {
        MIC_OPEN_FAILED
    }
}

/// Converts a signed 16-bit sample to `f32` in `[-1, 1]`.
pub fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / 32_768.0
}

/// Converts an unsigned 16-bit sample to `f32` in `[-1, 1]`.
pub fn u16_to_f32(sample: u16) -> f32 {
    (sample as f32 - 32_768.0) / 32_768.0
}

/// Converts `f32` in `[-1, 1]` to a signed 16-bit sample, clamping first.
pub fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * 32_767.0).round() as i16
}

/// Converts `f32` in `[-1, 1]` to an unsigned 16-bit sample, clamping first.
pub fn f32_to_u16(sample: f32) -> u16 {
    ((sample.clamp(-1.0, 1.0) * 32_767.0).round() as i32 + 32_768) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_samples_convert_at_the_ends_and_the_midpoint() {
        assert_eq!(i16_to_f32(i16::MIN), -1.0);
        assert_eq!(i16_to_f32(0), 0.0);
        assert!((i16_to_f32(i16::MAX) - 0.999_969_5).abs() < 1e-6);

        assert_eq!(f32_to_i16(-1.0), -32_767);
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), 32_767);
    }

    #[test]
    fn unsigned_samples_convert_at_the_ends_and_the_midpoint() {
        assert_eq!(u16_to_f32(0), -1.0);
        assert_eq!(u16_to_f32(32_768), 0.0);
        assert!((u16_to_f32(u16::MAX) - 0.999_969_5).abs() < 1e-6);

        assert_eq!(f32_to_u16(-1.0), 1);
        assert_eq!(f32_to_u16(0.0), 32_768);
        assert_eq!(f32_to_u16(1.0), u16::MAX);
    }

    #[test]
    fn conversion_clamps_out_of_range_values() {
        assert_eq!(f32_to_i16(2.0), 32_767);
        assert_eq!(f32_to_i16(-2.0), -32_767);
        assert_eq!(f32_to_u16(2.0), u16::MAX);
        assert_eq!(f32_to_u16(-2.0), 1);
    }

    #[test]
    fn permission_errors_map_to_the_denied_message() {
        assert_eq!(mic_error_message("Permission denied"), MIC_DENIED);
        assert_eq!(mic_error_message("access denied by policy"), MIC_DENIED);
        assert_eq!(mic_error_message("Not Authorized"), MIC_DENIED);
    }

    #[test]
    fn other_errors_map_to_the_open_failed_message() {
        assert_eq!(mic_error_message("device busy"), MIC_OPEN_FAILED);
        assert_eq!(mic_error_message(""), MIC_OPEN_FAILED);
    }

    #[cfg(not(feature = "audio"))]
    #[test]
    fn factories_report_missing_audio_support() {
        use regent_core::voice::{AudioOutFactory, MicCallbacks, MicFactory};

        let factory: MicFactory = mic_factory();
        let callbacks = MicCallbacks {
            on_samples: Box::new(|_: &[f32]| {}),
            on_error: Box::new(|_: String| {}),
        };
        match factory(callbacks) {
            Ok(_) => panic!("a build without audio must not open a microphone"),
            Err(message) => assert_eq!(message, NO_AUDIO),
        }

        let factory: AudioOutFactory = audio_out_factory();
        match factory() {
            Ok(_) => panic!("a build without audio must not open an output device"),
            Err(message) => assert_eq!(message, NO_AUDIO),
        }
    }
}
