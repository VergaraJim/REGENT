//! Speech-to-text through the bundled `regent-stt` helper process.
//!
//! The helper owns whisper.cpp so the app never links a second ggml next to
//! llama.cpp (decision 0005). [`client::HelperStt`] supervises it: it waits for
//! `ready`, sends one frame per utterance, restarts it once on failure, and
//! stops it on drop. [`wire`] re-implements the tiny protocol without making
//! `regent-stt` a dependency.

pub mod client;
pub mod process;
pub mod wire;

pub use client::stt_factory;
pub use process::{resolve_helper, HELPER_MISSING};
