//! Regent's pure logic lives here: no Tauri, no native engines.
//! Its tests run without models, audio hardware or a GPU.

pub mod chat;
pub mod context;
pub mod estimate;
pub mod footprint;
pub mod gguf;
pub mod hardware;
mod hash;
pub mod memory;
pub mod paths;
pub mod settings;
pub mod vision;
pub mod voice;
