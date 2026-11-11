//! The chat engine abstraction: everything the app needs from a local text
//! model, plus a deterministic [`FakeEngine`] for tests.
//!
//! Nothing here touches llama.cpp or Tauri. The real engine is adapted to
//! [`ChatEngine`] in a later stage; everything above the trait is tested with
//! the fake.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::prefix_cache::CacheIdentity;
use super::prompt::PromptMessage;

/// How to load the text model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    /// Path to the model file.
    pub model_path: PathBuf,
    /// Layers offloaded to the GPU.
    pub gpu_layers: u32,
    /// Context window in tokens.
    pub context_size: u32,
}

/// What one generation cost.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GenerationStats {
    /// Tokens the prompt took.
    pub prompt_tokens: usize,
    /// Prompt tokens served from the engine's prefix cache.
    pub reused_tokens: usize,
    /// Prompt tokens the engine had to process itself.
    pub processed_tokens: usize,
    /// Tokens the reply took.
    pub generated_tokens: usize,
    /// Decoding speed, in tokens per second.
    pub tokens_per_second: f32,
}

/// How a generation ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The model produced a complete reply.
    Finished(GenerationStats),
    /// The reply was stopped part way through.
    Cancelled(GenerationStats),
}

/// Why the engine could not load or generate.
///
/// The message is already written for the person using Regent: no jargon, no
/// error codes.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ChatEngineError(pub String);

/// The local text model as the chat service sees it.
pub trait ChatEngine: Send {
    /// Counts how many tokens `text` takes in this model.
    fn count_tokens(&self, text: &str) -> usize;

    /// The context window size, in tokens.
    fn context_tokens(&self) -> usize;

    /// Generates a reply to `prompt`, calling `on_token` for each piece of
    /// text as it is produced.
    ///
    /// The engine checks `cancel` between pieces: when it is set, generation
    /// stops and [`Outcome::Cancelled`] is returned with whatever was produced
    /// so far.
    fn generate(
        &mut self,
        prompt: &[PromptMessage],
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Outcome, ChatEngineError>;

    /// Writes a summary of `prompt`, using at most `max_tokens` for the reply.
    ///
    /// `Ok(None)` means the pass was cancelled; an error means the engine could
    /// not write a summary at all. Compaction treats both as "no summary this
    /// time" and leaves the prompt builder's fallback trimming to cope.
    fn summarize(
        &mut self,
        prompt: &[PromptMessage],
        max_tokens: usize,
        cancel: &AtomicBool,
    ) -> Result<Option<String>, ChatEngineError> {
        let _ = (prompt, max_tokens, cancel);
        Err(ChatEngineError(
            "This engine can't write summaries.".to_string(),
        ))
    }

    /// What the persisted prefix cache must match for this engine, or `None`
    /// when the engine cannot persist one.
    fn cache_identity(&self) -> Option<CacheIdentity> {
        None
    }

    /// Writes the KV state for `prefix` to `path` so a later session can load
    /// it instead of decoding the prefix again.
    fn save_prefix(
        &mut self,
        prefix: &[PromptMessage],
        path: &Path,
    ) -> Result<(), ChatEngineError> {
        let _ = (prefix, path);
        Err(ChatEngineError(
            "This engine can't save its prompt cache.".to_string(),
        ))
    }

    /// Loads the KV state for `prefix` from `path`.
    ///
    /// Returns whether the state was loaded and is now in use.
    fn load_prefix(
        &mut self,
        prefix: &[PromptMessage],
        path: &Path,
    ) -> Result<bool, ChatEngineError> {
        let _ = (prefix, path);
        Ok(false)
    }
}

/// A deterministic engine that emits fixed pieces.
///
/// It is public (not test-only) so other crates can build chat services
/// without a real model.
pub struct FakeEngine {
    pieces: Vec<String>,
    context_tokens: usize,
    fail: Option<String>,
    summaries: Vec<String>,
    prompts: Option<Arc<Mutex<Vec<Vec<PromptMessage>>>>>,
    prefix_cache: Option<Arc<Mutex<Vec<String>>>>,
}

impl FakeEngine {
    /// An engine that emits `pieces` one at a time, with a window of
    /// `context_tokens` tokens.
    pub fn new(pieces: Vec<String>, context_tokens: usize) -> FakeEngine {
        FakeEngine {
            pieces,
            context_tokens,
            fail: None,
            summaries: Vec::new(),
            prompts: None,
            prefix_cache: None,
        }
    }

    /// An engine whose every [`ChatEngine::generate`] fails with `message`.
    pub fn failing(message: impl Into<String>) -> FakeEngine {
        FakeEngine {
            pieces: Vec::new(),
            context_tokens: 4096,
            fail: Some(message.into()),
            summaries: Vec::new(),
            prompts: None,
            prefix_cache: None,
        }
    }

    /// The summaries this engine hands out, one per [`ChatEngine::summarize`]
    /// call, in order.
    pub fn with_summaries(mut self, summaries: Vec<String>) -> Self {
        self.summaries = summaries;
        self
    }

    /// Records every prompt passed to [`ChatEngine::generate`] or
    /// [`ChatEngine::summarize`] into `log`, oldest first.
    pub fn with_prompt_log(mut self, log: Arc<Mutex<Vec<Vec<PromptMessage>>>>) -> Self {
        self.prompts = Some(log);
        self
    }

    /// Backs [`ChatEngine::save_prefix`] and [`ChatEngine::load_prefix`] with a
    /// file, recording `"save"`, `"load:true"` or `"load:false"` in `log` so a
    /// test can see what the service did.
    pub fn with_prefix_cache(mut self, log: Arc<Mutex<Vec<String>>>) -> Self {
        self.prefix_cache = Some(log);
        self
    }

    /// Appends `entry` to the prefix cache log, when one is attached.
    fn log_prefix(&self, entry: &str) {
        if let Some(log) = &self.prefix_cache {
            log.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(entry.to_string());
        }
    }

    /// Records `prompt` when a prompt log is attached.
    fn record(&self, prompt: &[PromptMessage]) {
        if let Some(log) = &self.prompts {
            log.lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(prompt.to_vec());
        }
    }
}

impl ChatEngine for FakeEngine {
    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn context_tokens(&self) -> usize {
        self.context_tokens
    }

    fn generate(
        &mut self,
        prompt: &[PromptMessage],
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Outcome, ChatEngineError> {
        if let Some(message) = &self.fail {
            return Err(ChatEngineError(message.clone()));
        }

        self.record(prompt);

        let mut generated = String::new();
        for piece in &self.pieces {
            if cancel.load(Ordering::SeqCst) {
                return Ok(Outcome::Cancelled(stats_for(prompt, &generated)));
            }
            on_token(piece);
            generated.push_str(piece);
        }
        Ok(Outcome::Finished(stats_for(prompt, &generated)))
    }

    fn summarize(
        &mut self,
        prompt: &[PromptMessage],
        _max_tokens: usize,
        cancel: &AtomicBool,
    ) -> Result<Option<String>, ChatEngineError> {
        self.record(prompt);
        if cancel.load(Ordering::SeqCst) {
            return Ok(None);
        }
        if self.summaries.is_empty() {
            return Err(ChatEngineError(
                "This engine has no summary to give.".to_string(),
            ));
        }
        Ok(Some(self.summaries.remove(0)))
    }

    fn cache_identity(&self) -> Option<CacheIdentity> {
        self.prefix_cache.as_ref().map(|_| CacheIdentity {
            kv_type: "f16".to_string(),
            chat_template: "fake-template".to_string(),
            engine_version: "fake-1".to_string(),
            context_size: self.context_tokens,
        })
    }

    fn save_prefix(
        &mut self,
        prefix: &[PromptMessage],
        path: &Path,
    ) -> Result<(), ChatEngineError> {
        let Some(message) = prefix.first() else {
            return Err(ChatEngineError("There is no prefix to save.".to_string()));
        };
        std::fs::write(path, &message.text)
            .map_err(|error| ChatEngineError(format!("Couldn't save the prompt cache: {error}")))?;
        self.log_prefix("save");
        Ok(())
    }

    fn load_prefix(
        &mut self,
        prefix: &[PromptMessage],
        path: &Path,
    ) -> Result<bool, ChatEngineError> {
        let saved = std::fs::read_to_string(path).unwrap_or_default();
        let loaded = prefix.first().is_some_and(|message| message.text == saved);
        self.log_prefix(if loaded { "load:true" } else { "load:false" });
        Ok(loaded)
    }
}

/// The stats for a reply, counted the same way as [`FakeEngine::count_tokens`].
fn stats_for(prompt: &[PromptMessage], generated: &str) -> GenerationStats {
    let prompt_tokens: usize = prompt
        .iter()
        .map(|message| message.text.split_whitespace().count())
        .sum();
    GenerationStats {
        prompt_tokens,
        // The fake engine has no prefix cache.
        reused_tokens: 0,
        processed_tokens: prompt_tokens,
        generated_tokens: generated.split_whitespace().count(),
        // The fake engine has no timing to report; the real engine fills this.
        tokens_per_second: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::prompt::{PromptRole, SYSTEM_PROMPT};

    fn prompt() -> Vec<PromptMessage> {
        vec![PromptMessage {
            role: PromptRole::System,
            text: SYSTEM_PROMPT.to_string(),
        }]
    }

    fn pieces(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

    #[test]
    fn count_tokens_counts_words() {
        let engine = FakeEngine::new(Vec::new(), 4096);
        assert_eq!(engine.count_tokens("one two three"), 3);
        assert_eq!(engine.count_tokens("   "), 0);
        assert_eq!(engine.context_tokens(), 4096);
    }

    #[test]
    fn generate_emits_every_piece_and_reports_stats() {
        let mut engine = FakeEngine::new(pieces(&["Hello", ", ", "world"]), 4096);
        let cancel = AtomicBool::new(false);
        let mut seen = String::new();

        let outcome = engine
            .generate(&prompt(), &cancel, &mut |piece| seen.push_str(piece))
            .expect("generate");

        assert_eq!(seen, "Hello, world");
        let Outcome::Finished(stats) = outcome else {
            panic!("expected a finished reply");
        };
        assert_eq!(stats.generated_tokens, 2);
        assert_eq!(
            stats.prompt_tokens,
            SYSTEM_PROMPT.split_whitespace().count()
        );
    }

    #[test]
    fn a_set_cancel_flag_stops_before_the_next_piece() {
        let mut engine = FakeEngine::new(pieces(&["one", "two", "three"]), 4096);
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();

        let outcome = engine
            .generate(&prompt(), &cancel, &mut |piece| {
                seen.push(piece.to_string());
                cancel.store(true, Ordering::SeqCst);
            })
            .expect("generate");

        assert_eq!(seen, vec!["one"]);
        assert!(matches!(outcome, Outcome::Cancelled(_)));
    }

    #[test]
    fn a_failing_engine_returns_its_message() {
        let mut engine = FakeEngine::failing("The model stopped unexpectedly.");
        let cancel = AtomicBool::new(false);
        let error = engine
            .generate(&prompt(), &cancel, &mut |_| {})
            .expect_err("generate should fail");
        assert_eq!(error.to_string(), "The model stopped unexpectedly.");
    }
}
