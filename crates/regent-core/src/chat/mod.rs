//! Chat state: the message model, the on-disk history store, the prompt
//! builder, the engine abstraction and the service that ties them together.
//!
//! Everything here is pure logic: the real text model is adapted to
//! [`engine::ChatEngine`] by a later stage.

pub mod compaction;
pub mod engine;
pub mod message;
pub mod prefix_cache;
pub mod prompt;
pub mod service;
pub mod store;
pub mod think;
pub mod tools;

pub use compaction::{
    accept_summary, build_summary_prompt, plan_compaction, summarizer_budget, summary_cap,
    working_memory_tokens, CompactionPlan, KEEP_VERBATIM, MAX_COMPACTION_PASSES,
    MESSAGE_TEMPLATE_COST, MIN_SUMMARY_TOKENS, SUMMARIZER_SYSTEM_PROMPT, SUMMARY_HEADING,
    SUMMARY_TEMPERATURE,
};
pub use engine::{ChatEngine, ChatEngineError, EngineConfig, FakeEngine, GenerationStats, Outcome};
pub use message::{ChatMessage, Role};
pub use prefix_cache::{CacheIdentity, PrefixCache, PrefixKey};
pub use prompt::{
    build_prompt, fixed_prefix, system_text, PromptMessage, PromptRole, REPLY_RESERVE_TOKENS,
    SYSTEM_PROMPT,
};
pub use service::{
    ChatEvent, ChatService, ChatServiceDeps, ChatState, ChatStatus, EngineFactory, StatusSink,
    TurnSettings,
};
pub use store::{ChatStore, ChatStoreError, Summary, HISTORY_READ_CAP};
pub use think::{strip_think, ThinkFilter, THINK_CLOSE, THINK_OPEN};
pub use tools::{
    look_at_screen_spec, search_memory_spec, tools_section, validate_call, ParamKind, ParamSpec,
    ParsedPiece, ToolCallParser, ToolSpec, ValidCall, ANSWER_NOW, MAX_TOOL_CALLS_PER_TURN,
    TOOL_CALL_CLOSE, TOOL_CALL_OPEN, TOOL_RESULT_HEADING,
};
