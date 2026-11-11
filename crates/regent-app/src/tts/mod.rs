//! Text-to-speech through the bundled Piper engine.
//!
//! Piper ships inside the app as a resource folder (decision 0006), so it is never
//! taken from `PATH` or a system install. It runs as a child process: the app writes
//! text to its stdin and reads raw little-endian `i16` samples from its stdout.

pub mod piper;
pub mod synth;

pub use piper::{
    piper_command, resolve_piper, synthesize_once, voice_config_path, PiperPaths, TtsError,
};
pub use synth::{synth_factory, PiperSynthesizer};
