//! Chat state that needs no model: the message model, the on-disk history
//! store, and the prompt builder.

pub mod message;
pub mod prompt;
pub mod store;

pub use message::{ChatMessage, Role};
pub use prompt::{build_prompt, PromptMessage, PromptRole, REPLY_RESERVE_TOKENS, SYSTEM_PROMPT};
pub use store::{ChatStore, ChatStoreError, HISTORY_READ_CAP};
