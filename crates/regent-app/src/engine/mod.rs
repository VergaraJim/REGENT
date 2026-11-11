//! Local inference inside the app process.
//!
//! Text, vision and embeddings run on llama.cpp (via `llama-cpp-2`), behind the
//! default `llama` feature of `regent-app`. Speech-to-text deliberately lives in
//! the separate `regent-stt` helper instead (see decision 0004).

#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "llama")]
pub mod llama_chat;
#[cfg(feature = "llama")]
pub mod llama_embed;

#[cfg(feature = "llama")]
pub use llama::{smoke_generate, EngineError, SmokeOutput};
#[cfg(feature = "llama")]
pub use llama_chat::LlamaChatEngine;
#[cfg(feature = "llama")]
pub use llama_embed::LlamaEmbeddingEngine;
