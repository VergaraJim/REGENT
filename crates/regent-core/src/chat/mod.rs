//! Chat state: the message model, the on-disk history store, the prompt
//! builder, the engine abstraction and the service that ties them together.
//!
//! Everything here is pure logic: the real text model is adapted to
//! [`engine::ChatEngine`] by a later stage.

pub mod compaction;
pub mod engine;
pub mod message;
pub mod prompt;
pub mod service;
pub mod store;

pub use compaction::{
    accept_summary, build_summary_prompt, plan_compaction, summarizer_budget, summary_cap,
    working_memory_tokens, CompactionPlan, KEEP_VERBATIM, MAX_COMPACTION_PASSES,
    MESSAGE_TEMPLATE_COST, MIN_SUMMARY_TOKENS, SUMMARIZER_SYSTEM_PROMPT, SUMMARY_HEADING,
    SUMMARY_TEMPERATURE,
};
pub use engine::{ChatEngine, ChatEngineError, EngineConfig, FakeEngine, GenerationStats, Outcome};
pub use message::{ChatMessage, Role};
pub use prompt::{
    build_prompt, system_text, PromptMessage, PromptRole, REPLY_RESERVE_TOKENS, SYSTEM_PROMPT,
};
pub use service::{ChatEvent, ChatService, ChatState, ChatStatus, EngineFactory, StatusSink};
pub use store::{ChatStore, ChatStoreError, Summary, HISTORY_READ_CAP};
