//! Speech-to-text helper process for Regent.
//!
//! The library half is the pipe protocol and the request loop. It has no
//! whisper dependency, so it is fully testable with a fake transcriber. The
//! binary half (`src/main.rs` plus `src/whisper.rs`) is the only place that
//! links whisper.cpp, and it sits behind the default `whisper` feature.

pub mod protocol;
pub mod service;

#[cfg(feature = "whisper")]
pub mod whisper;
