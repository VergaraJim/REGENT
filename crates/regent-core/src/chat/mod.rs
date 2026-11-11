//! Chat state: the message model, the on-disk history store, the prompt
//! builder, the engine abstraction and the service that ties them together.
//!
//! Everything here is pure logic: the real text model is adapted to
//! [`engine::ChatEngine`] by a later stage.

pub mod engine;
pub mod message;
pub mod prompt;
pub mod service;
pub mod store;

pub use engine::{ChatEngine, ChatEngineError, EngineConfig, FakeEngine, GenerationStats, Outcome};
pub use message::{ChatMessage, Role};
pub use prompt::{build_prompt, PromptMessage, PromptRole, REPLY_RESERVE_TOKENS, SYSTEM_PROMPT};
pub use service::{ChatEvent, ChatService, ChatState, ChatStatus, EngineFactory, StatusSink};
pub use store::{ChatStore, ChatStoreError, HISTORY_READ_CAP};
