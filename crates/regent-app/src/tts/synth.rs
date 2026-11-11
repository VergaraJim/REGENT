//! The Piper-backed [`Synthesizer`]: one sentence in, mono `f32` samples out.
//!
//! Loading resolves the bundled engine and reads the voice's `.onnx.json` once,
//! so the sample rate each sentence is played at is known without reopening the
//! config. Speaking a sentence runs the engine once (decision 0006).

use std::fs;
use std::path::{Path, PathBuf};

use regent_core::voice::{SynthFactory, Synthesizer};

use crate::tts::piper::{resolve_piper, synthesize_once, voice_config_path, PiperPaths, TtsError};

/// The rate assumed when a voice config does not name one.
pub const DEFAULT_SAMPLE_RATE: u32 = 22_050;

/// Shown when the picked voice file is gone.
pub const VOICE_MISSING: &str = "The voice file couldn't be found. Choose it again in Settings.";

/// Shown when the voice's `.onnx.json` cannot be read.
pub const VOICE_CONFIG_UNREADABLE: &str = "This voice's .onnx.json file couldn't be read.";

/// Shown when one sentence could not be synthesized.
pub const SENTENCE_FAILED: &str = "Regent couldn't speak part of this reply.";

/// Reads the voice's sample rate out of its `.onnx.json` text.
///
/// A missing, unusable or zero `audio.sample_rate` falls back to
/// [`DEFAULT_SAMPLE_RATE`]; only text that is not JSON at all is an error.
pub fn voice_sample_rate(config_json: &str) -> Result<u32, String> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|_| VOICE_CONFIG_UNREADABLE.to_string())?;

    let rate = config
        .get("audio")
        .and_then(|audio| audio.get("sample_rate"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|rate| u32::try_from(rate).ok())
        .unwrap_or(DEFAULT_SAMPLE_RATE);

    Ok(if rate == 0 { DEFAULT_SAMPLE_RATE } else { rate })
}

/// Converts one raw Piper sample to the `[-1, 1]` range playback wants.
fn sample_to_f32(sample: i16) -> f32 {
    f32::from(sample) / 32_768.0
}

/// A synthesizer that speaks through the bundled Piper engine.
#[derive(Debug)]
pub struct PiperSynthesizer {
    paths: PiperPaths,
    voice: PathBuf,
    sample_rate: u32,
}

impl PiperSynthesizer {
    /// Resolves the engine and reads `voice`'s config, ready to speak.
    pub fn load(resource_dir: &Path, voice: &Path) -> Result<Self, String> {
        let paths = resolve_piper(resource_dir).map_err(|error| error.to_string())?;

        if !voice.is_file() {
            return Err(VOICE_MISSING.to_string());
        }

        let config_path = voice_config_path(voice).map_err(|error| error.to_string())?;
        let config_json =
            fs::read_to_string(&config_path).map_err(|_| VOICE_CONFIG_UNREADABLE.to_string())?;
        let sample_rate = voice_sample_rate(&config_json)?;

        Ok(Self {
            paths,
            voice: voice.to_path_buf(),
            sample_rate,
        })
    }

    /// The rate the synthesized samples are in.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Synthesizer for PiperSynthesizer {
    fn synthesize(&mut self, sentence: &str) -> Result<(Vec<f32>, u32), String> {
        let raw = synthesize_once(&self.paths, &self.voice, sentence).map_err(|error| {
            eprintln!("regent: piper failed: {error}");
            SENTENCE_FAILED.to_string()
        })?;

        if raw.is_empty() {
            return Err(SENTENCE_FAILED.to_string());
        }

        let samples = raw.into_iter().map(sample_to_f32).collect();
        Ok((samples, self.sample_rate))
    }
}

/// Builds the [`SynthFactory`] that speaks through the bundled engine.
///
/// `resource_dir` is where the app resolved its resources; `None` means that
/// lookup already failed, so every load reports a missing engine.
pub fn synth_factory(resource_dir: Option<PathBuf>) -> SynthFactory {
    Box::new(move |voice: &Path| {
        let Some(dir) = resource_dir.as_deref() else {
            return Err(TtsError::EngineMissing.to_string());
        };
        let synthesizer = PiperSynthesizer::load(dir, voice)?;
        Ok(Box::new(synthesizer) as Box<dyn Synthesizer>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::piper::EXE_SUFFIX;

    /// The layout the fetch script installs, with no voice in it.
    fn resource_with_engine() -> tempfile::TempDir {
        let resource_dir = tempfile::tempdir().expect("temp dir");
        let piper_dir = resource_dir.path().join("piper");
        fs::create_dir_all(piper_dir.join("espeak-ng-data")).expect("espeak data dir");
        fs::write(piper_dir.join(format!("piper{EXE_SUFFIX}")), b"").expect("write exe");
        resource_dir
    }

    fn write_voice(dir: &Path, config: Option<&str>) -> PathBuf {
        let voice = dir.join("en_US-lessac-low.onnx");
        fs::write(&voice, b"not a real model").expect("write voice");
        if let Some(config) = config {
            let mut raw = voice.as_os_str().to_os_string();
            raw.push(".json");
            fs::write(PathBuf::from(raw), config.as_bytes()).expect("write config");
        }
        voice
    }

    #[test]
    fn the_config_sample_rate_is_read() {
        assert_eq!(
            voice_sample_rate(r#"{"audio":{"sample_rate":16000}}"#),
            Ok(16_000)
        );
    }

    #[test]
    fn a_missing_sample_rate_falls_back_to_the_default() {
        assert_eq!(voice_sample_rate("{}"), Ok(DEFAULT_SAMPLE_RATE));
        assert_eq!(
            voice_sample_rate(r#"{"audio":{"sample_rate":0}}"#),
            Ok(DEFAULT_SAMPLE_RATE)
        );
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert_eq!(
            voice_sample_rate("not json"),
            Err(VOICE_CONFIG_UNREADABLE.to_string())
        );
    }

    #[test]
    fn raw_samples_are_scaled_to_the_unit_range() {
        assert_eq!(sample_to_f32(-32_768), -1.0);
        assert_eq!(sample_to_f32(0), 0.0);
        assert_eq!(sample_to_f32(32_767), 32_767.0 / 32_768.0);
    }

    #[test]
    fn a_missing_engine_is_reported() {
        let resource_dir = tempfile::tempdir().expect("temp dir");
        let voice = write_voice(resource_dir.path(), Some("{}"));

        let error = PiperSynthesizer::load(resource_dir.path(), &voice)
            .expect_err("the engine should be missing");

        assert_eq!(error, TtsError::EngineMissing.to_string());
    }

    #[test]
    fn a_missing_voice_is_reported() {
        let resource_dir = resource_with_engine();
        let missing = resource_dir.path().join("en_US-lessac-low.onnx");

        let error = PiperSynthesizer::load(resource_dir.path(), &missing)
            .expect_err("voice should be gone");

        assert_eq!(error, VOICE_MISSING);
    }

    #[test]
    fn a_voice_without_its_config_is_reported() {
        let resource_dir = resource_with_engine();
        let voice = write_voice(resource_dir.path(), None);

        let error = PiperSynthesizer::load(resource_dir.path(), &voice)
            .expect_err("config should be missing");

        assert_eq!(error, TtsError::VoiceConfigMissing.to_string());
    }

    #[test]
    fn a_valid_voice_loads_with_its_sample_rate() {
        let resource_dir = resource_with_engine();
        let voice = write_voice(
            resource_dir.path(),
            Some(r#"{"audio":{"sample_rate":16000}}"#),
        );

        let synthesizer =
            PiperSynthesizer::load(resource_dir.path(), &voice).expect("voice should load");

        assert_eq!(synthesizer.sample_rate(), 16_000);
    }

    #[test]
    fn an_unreadable_config_is_reported() {
        let resource_dir = resource_with_engine();
        let voice = write_voice(resource_dir.path(), None);
        // A config that is not UTF-8 cannot be read as text.
        let mut raw = voice.as_os_str().to_os_string();
        raw.push(".json");
        fs::write(PathBuf::from(raw), [0xff, 0xfe, 0x00]).expect("write config");

        let error = PiperSynthesizer::load(resource_dir.path(), &voice)
            .expect_err("config should be unreadable");

        assert_eq!(error, VOICE_CONFIG_UNREADABLE);
    }

    #[test]
    fn a_missing_resource_dir_makes_the_factory_fail() {
        let factory = synth_factory(None);

        let error = factory(Path::new("/voices/test.onnx"))
            .err()
            .expect("the factory should fail");

        assert_eq!(error, TtsError::EngineMissing.to_string());
    }
}
