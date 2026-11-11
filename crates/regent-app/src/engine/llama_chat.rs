//! The llama.cpp implementation of [`ChatEngine`].
//!
//! The engine keeps one model and one context alive for its whole life. Every
//! turn renders the prompt with the model's own chat template (falling back to
//! ChatML), compares the new prompt tokens with the ones already in the KV
//! cache, drops the cache after the shared prefix and decodes only the rest.
//! Summaries run on a temporary second context so they never evict that cache.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use regent_core::chat::SUMMARY_TEMPERATURE;
use regent_core::chat::{
    CacheIdentity, ChatEngine, ChatEngineError, EngineConfig, GenerationStats, Outcome,
    PromptMessage, PromptRole, REPLY_RESERVE_TOKENS,
};
use regent_core::context::{clamp_requested, context_limits, MIN_CONTEXT};

use super::llama::backend;

/// Bump this whenever the `llama-cpp-2` crate version changes: the KV state
/// format is tied to it, so a saved state must not outlive the version.
const ENGINE_VERSION: &str = "llama-cpp-2 0.1.161";

/// The fixed seed keeps a replay of the same prompt reproducible.
const SAMPLE_SEED: u32 = 0x5EED_1234;

/// Sampling: keep the 40 best tokens, then the smallest set covering 95% of
/// the probability, then apply the temperature, then draw.
const TOP_K: i32 = 40;
const TOP_P: f32 = 0.95;
const TEMPERATURE: f32 = 0.7;

/// The step a summarizer context size is rounded up to.
const CONTEXT_STEP: usize = 256;

/// Plain-language message for a summarizer prompt that does not fit the window.
const SUMMARY_TOO_LONG: &str = "This conversation is too long to summarize.";

/// Plain-language messages for every failure.
const BACKEND_FAILED: &str =
    "The text engine couldn't start on this PC. Restart Regent and try again.";
const LOAD_FAILED: &str = "The model couldn't be loaded. It may be damaged or too big for this PC.";
const CONTEXT_FAILED: &str =
    "The model couldn't start on this PC. Try a smaller context in Settings.";
const EMPTY_PROMPT: &str = "There was nothing to send to the model.";
const TOO_LONG: &str = "This conversation is too long for the model. Start a new chat.";
const DECODE_FAILED: &str = "The model stopped unexpectedly. Please try again.";

/// A local text model served by llama.cpp.
pub struct LlamaChatEngine {
    /// Borrows `_model`; declared first so it is dropped before the model.
    context: LlamaContext<'static>,
    /// Owns the model; kept only so the context stays valid and is freed last.
    _model: Box<LlamaModel>,
    /// The context window that was actually loaded.
    context_size: usize,
    /// The tokens whose KV entries are in sequence 0, in position order.
    cached: Vec<LlamaToken>,
}

impl LlamaChatEngine {
    /// Loads the model and creates the single context used for every turn.
    pub fn load(config: &EngineConfig) -> Result<Self, ChatEngineError> {
        let backend = backend().map_err(|_| ChatEngineError(BACKEND_FAILED.to_string()))?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
        let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)
            .map_err(|_| ChatEngineError(LOAD_FAILED.to_string()))?;
        let model = Box::new(model);

        // The requested context is clamped against what the model was trained
        // for (and rounded to the context step) by `regent-core`.
        let train_context = model.n_ctx_train().max(MIN_CONTEXT);
        let limits = context_limits(Some(u64::from(train_context)), None);
        let context_size = clamp_requested(config.context_size, &limits)
            .value
            .unwrap_or(MIN_CONTEXT);

        let n_ctx = NonZeroU32::new(context_size)
            .ok_or_else(|| ChatEngineError(CONTEXT_FAILED.to_string()))?;
        let context_params = context_params(n_ctx, thread_count());

        // SAFETY: `model` is boxed, so its address is stable across the move
        // into `Self`. The `context` field is declared before `_model`, so the
        // context is always dropped first and never outlives the model.
        let model_ptr: *const LlamaModel = &*model;
        let model_ref: &'static LlamaModel = unsafe { &*model_ptr };
        let context = model_ref
            .new_context(backend, context_params)
            .map_err(|_| ChatEngineError(CONTEXT_FAILED.to_string()))?;

        Ok(Self {
            context,
            _model: model,
            context_size: usize::try_from(context_size).unwrap_or(usize::MAX),
            cached: Vec::new(),
        })
    }

    /// Renders the ordered messages into one prompt string.
    ///
    /// The model's own template is preferred; when the model has none, or
    /// rendering fails, the widely supported ChatML layout is used.
    fn render_prompt(&self, prompt: &[PromptMessage]) -> String {
        if let Some(rendered) = self.render_with_template(prompt) {
            return rendered;
        }
        chatml_prompt(prompt)
    }

    /// Renders with the template stored in the model, if there is one.
    fn render_with_template(&self, prompt: &[PromptMessage]) -> Option<String> {
        let template = self.model().chat_template(None).ok()?;
        let messages: Vec<LlamaChatMessage> = prompt
            .iter()
            .map(|message| {
                LlamaChatMessage::new(role_name(message.role).to_string(), message.text.clone())
                    .ok()
            })
            .collect::<Option<Vec<_>>>()?;
        self.model()
            .apply_chat_template(&template, &messages, true)
            .ok()
            .filter(|text| !text.is_empty())
    }

    /// Tokenises a rendered prompt, adding BOS only when the template did not
    /// already supply it.
    fn tokenize_prompt(&self, text: &str) -> Vec<LlamaToken> {
        let vocab = self.model().vocab();
        let mut tokens = vocab.tokenize(text.as_bytes(), false, true);
        let starts_with_bos = tokens.first().is_some_and(|token| *token == vocab.bos());
        if !starts_with_bos && vocab.should_add_bos() {
            tokens.insert(0, vocab.bos());
        }
        tokens
    }

    /// The model behind the context.
    fn model(&self) -> &LlamaModel {
        &self._model
    }

    /// The fixed opening of `prefix` as model tokens, or an empty vector when
    /// it cannot be told apart from the rest of the prompt.
    ///
    /// Some templates merge the system text into the first user turn, so
    /// rendering the prefix alone is not reliable. Rendering it with two
    /// different user turns instead and keeping what both renderings share
    /// isolates the tokens that belong to the prefix.
    fn prefix_tokens(&self, prefix: &[PromptMessage]) -> Vec<LlamaToken> {
        let with_user = |text: &str| {
            let mut messages = prefix.to_vec();
            messages.push(PromptMessage {
                role: PromptRole::User,
                text: text.to_string(),
            });
            self.tokenize_prompt(&self.render_prompt(&messages))
        };
        let first = with_user("a");
        let second = with_user("b");
        let shared = common_prefix_len(&first, &second);
        first[..shared].to_vec()
    }
}

impl ChatEngine for LlamaChatEngine {
    fn count_tokens(&self, text: &str) -> usize {
        self.model()
            .vocab()
            .tokenize(text.as_bytes(), false, false)
            .len()
    }

    fn context_tokens(&self) -> usize {
        self.context_size
    }

    fn cache_identity(&self) -> Option<CacheIdentity> {
        let chat_template = self
            .model()
            .chat_template(None)
            .ok()
            .and_then(|template| template.to_str().ok().map(str::to_string))
            .unwrap_or_else(|| "chatml-fallback".to_string());
        Some(CacheIdentity {
            // The context keeps llama.cpp's default KV cache type, f16.
            kv_type: "f16".to_string(),
            chat_template,
            engine_version: ENGINE_VERSION.to_string(),
            context_size: self.context_size,
        })
    }

    /// Saves the KV state for the fixed `prefix` to `path`.
    ///
    /// The cache is trimmed to the prefix the first time this runs, so the
    /// tokens decoded after the prefix have to be decoded once more on the
    /// next turn; that trim happens at most once per saved state.
    fn save_prefix(
        &mut self,
        prefix: &[PromptMessage],
        path: &Path,
    ) -> Result<(), ChatEngineError> {
        let tokens = self.prefix_tokens(prefix);
        if tokens.is_empty() || !self.cached.starts_with(&tokens) {
            return Err(ChatEngineError(
                "The prompt cache isn't ready to save.".to_string(),
            ));
        }

        if self.cached.len() > tokens.len() {
            let trim = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
            if self.context.kv_cache_seq_rm(0, Some(trim), None).is_err() {
                return Err(ChatEngineError(
                    "The prompt cache isn't ready to save.".to_string(),
                ));
            }
            self.cached.truncate(tokens.len());
        }

        self.context
            .state_seq_save_file(path, 0, &tokens)
            .map_err(|_| ChatEngineError("Regent couldn't save its prompt cache.".to_string()))?;
        Ok(())
    }

    fn load_prefix(
        &mut self,
        prefix: &[PromptMessage],
        path: &Path,
    ) -> Result<bool, ChatEngineError> {
        let tokens = self.prefix_tokens(prefix);
        if tokens.is_empty() {
            return Ok(false);
        }

        self.context.clear_kv_cache();
        self.cached.clear();

        match self.context.state_seq_load_file(path, 0, self.context_size) {
            Ok((loaded, _)) if loaded == tokens => {
                self.cached = loaded;
                Ok(true)
            }
            // A bad or mismatched state file is never an error the user sees.
            _ => {
                self.context.clear_kv_cache();
                Ok(false)
            }
        }
    }

    fn generate(
        &mut self,
        prompt: &[PromptMessage],
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Outcome, ChatEngineError> {
        let rendered = self.render_prompt(prompt);

        let tokens = self.tokenize_prompt(&rendered);
        let prompt_len = tokens.len();
        if prompt_len == 0 {
            return Err(ChatEngineError(EMPTY_PROMPT.to_string()));
        }
        if prompt_len >= self.context_size {
            return Err(ChatEngineError(TOO_LONG.to_string()));
        }

        // Reuse the longest shared token prefix already in the cache. At least
        // one prompt token must be decoded again to produce logits.
        let mut reuse = reusable_len(common_prefix_len(&self.cached, &tokens), prompt_len);
        if reuse < self.cached.len()
            && self
                .context
                .kv_cache_seq_rm(0, Some(reuse as u32), None)
                .is_err()
        {
            // The model cannot drop part of its cache (sliding window or
            // recurrent attention): start over instead.
            self.context.clear_kv_cache();
            reuse = 0;
        }

        // Decode the prompt in chunks no larger than the context's batch size;
        // only the very last token computes logits.
        let batch_size = usize::try_from(self.context.n_batch()).unwrap_or(1).max(1);
        let decoded = decode_prompt(&mut self.context, &tokens[reuse..], reuse, batch_size);
        let (mut last_index, mut pos) = match decoded {
            Ok(decoded) => decoded,
            Err(error) => {
                self.context.clear_kv_cache();
                self.cached.clear();
                return Err(error);
            }
        };
        // Everything from `reuse` on has just been decoded, so the whole
        // prompt is now in the cache.
        self.cached = tokens.clone();

        let mut sampler = sampler(TEMPERATURE);

        let max_new = (self.context_size - prompt_len).min(REPLY_RESERVE_TOKENS);
        let started = Instant::now();
        let mut generated_tokens = 0usize;
        let mut pending: Vec<u8> = Vec::new();
        let mut cancelled = false;
        let mut batch = LlamaBatch::new(1, 1);

        while generated_tokens < max_new && (pos as usize) < self.context_size {
            if cancel.load(Ordering::SeqCst) {
                cancelled = true;
                break;
            }

            let token = sampler.sample(&self.context, last_index);
            if self.model().vocab().is_eog(token) {
                break;
            }
            sampler.accept(token);
            generated_tokens += 1;

            pending.extend(self.model().vocab().token_to_piece(token, false, None));
            flush_text(&mut pending, on_token);

            batch.clear();
            if batch.add(token, pos, &[0], true).is_err() {
                self.context.clear_kv_cache();
                self.cached.clear();
                return Err(ChatEngineError(DECODE_FAILED.to_string()));
            }
            if self.context.decode(&mut batch).is_err() {
                self.context.clear_kv_cache();
                self.cached.clear();
                return Err(ChatEngineError(DECODE_FAILED.to_string()));
            }
            self.cached.push(token);
            pos += 1;
            last_index = 0;
        }

        // A token can end in the middle of a character; never drop those bytes.
        if !pending.is_empty() {
            let rest = String::from_utf8_lossy(&pending);
            on_token(&rest);
        }

        let seconds = started.elapsed().as_secs_f64();
        let tokens_per_second = if seconds > 0.0 {
            generated_tokens as f32 / seconds as f32
        } else {
            0.0
        };
        let stats = GenerationStats {
            prompt_tokens: prompt_len,
            reused_tokens: reuse,
            processed_tokens: prompt_len - reuse,
            generated_tokens,
            tokens_per_second,
        };

        if cancelled {
            Ok(Outcome::Cancelled(stats))
        } else {
            Ok(Outcome::Finished(stats))
        }
    }

    fn summarize(
        &mut self,
        prompt: &[PromptMessage],
        max_tokens: usize,
        cancel: &AtomicBool,
    ) -> Result<Option<String>, ChatEngineError> {
        let rendered = self.render_prompt(prompt);
        let tokens = self.tokenize_prompt(&rendered);
        let prompt_len = tokens.len();
        if prompt_len == 0 {
            return Err(ChatEngineError(EMPTY_PROMPT.to_string()));
        }

        // A summarizer pass runs on its own temporary context, so the main
        // conversation's cache is never evicted.
        let Some(n_ctx) = summarizer_context_size(prompt_len, max_tokens, self.context_size) else {
            return Err(ChatEngineError(SUMMARY_TOO_LONG.to_string()));
        };
        let n_ctx =
            NonZeroU32::new(n_ctx).ok_or_else(|| ChatEngineError(CONTEXT_FAILED.to_string()))?;
        let backend = backend().map_err(|_| ChatEngineError(BACKEND_FAILED.to_string()))?;
        let mut context = self
            ._model
            .new_context(backend, context_params(n_ctx, thread_count()))
            .map_err(|_| ChatEngineError(CONTEXT_FAILED.to_string()))?;

        let batch_size = usize::try_from(context.n_batch()).unwrap_or(1).max(1);
        let (mut last_index, mut pos) = decode_prompt(&mut context, &tokens, 0, batch_size)?;
        let mut sampler = sampler(SUMMARY_TEMPERATURE);

        let mut text = String::new();
        let mut pending: Vec<u8> = Vec::new();
        let mut generated_tokens = 0usize;
        let mut batch = LlamaBatch::new(1, 1);

        while generated_tokens < max_tokens && (pos as usize) < n_ctx.get() as usize {
            if cancel.load(Ordering::SeqCst) {
                return Ok(None);
            }

            let token = sampler.sample(&context, last_index);
            if self.model().vocab().is_eog(token) {
                break;
            }
            sampler.accept(token);
            generated_tokens += 1;

            pending.extend(self.model().vocab().token_to_piece(token, false, None));
            flush_text(&mut pending, &mut |piece| text.push_str(piece));

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
            context
                .decode(&mut batch)
                .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
            pos += 1;
            last_index = 0;
        }

        // A token can end in the middle of a character; never drop those bytes.
        if !pending.is_empty() {
            text.push_str(&String::from_utf8_lossy(&pending));
        }

        drop(context);
        Ok(Some(text))
    }
}

/// How many leading items `a` and `b` share.
fn common_prefix_len<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(left, right)| left == right)
        .count()
}

/// How much of the prompt the cache can serve.
///
/// At least one prompt token must be decoded again to get logits.
fn reusable_len(common: usize, prompt_len: usize) -> usize {
    common.min(prompt_len.saturating_sub(1))
}

/// The context size for a summarizer pass, or `None` when it cannot fit.
///
/// It is the prompt, room for `max_tokens` and a small margin, rounded up to a
/// multiple of 256 and never smaller than 256.
fn summarizer_context_size(prompt_len: usize, max_tokens: usize, main_ctx: usize) -> Option<u32> {
    let needed = prompt_len.saturating_add(max_tokens).saturating_add(8);
    let rounded = needed.div_ceil(CONTEXT_STEP).saturating_mul(CONTEXT_STEP);
    let size = rounded.max(CONTEXT_STEP);
    if size > main_ctx {
        return None;
    }
    u32::try_from(size).ok()
}

/// The threads every context uses.
fn thread_count() -> i32 {
    i32::try_from(
        std::thread::available_parallelism()
            .map_or(1, |count| count.get())
            .saturating_sub(1)
            .max(1),
    )
    .unwrap_or(1)
}

/// Context parameters shared by the main and the summarizer contexts.
fn context_params(n_ctx: NonZeroU32, threads: i32) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
}

/// The sampler chain used for replies and summaries.
fn sampler(temperature: f32) -> LlamaSampler {
    LlamaSampler::chain_simple([
        LlamaSampler::top_k(TOP_K),
        LlamaSampler::top_p(TOP_P, 1),
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(SAMPLE_SEED),
    ])
}

/// Decodes `tokens` into `context`, starting at position `start_pos`.
///
/// Only the very last token computes logits. Returns that batch row and the
/// next free position.
fn decode_prompt(
    context: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    start_pos: usize,
    batch_size: usize,
) -> Result<(i32, i32), ChatEngineError> {
    let mut batch = LlamaBatch::new(batch_size.min(tokens.len()).max(1), 1);
    let mut pos = i32::try_from(start_pos).unwrap_or(i32::MAX);
    let mut last_index = 0i32;
    for (chunk_index, chunk) in tokens.chunks(batch_size).enumerate() {
        let chunk_start = chunk_index * batch_size;
        batch.clear();
        for (offset, &token) in chunk.iter().enumerate() {
            let is_last = chunk_start + offset + 1 == tokens.len();
            batch
                .add(token, pos, &[0], is_last)
                .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
            pos += 1;
        }
        context
            .decode(&mut batch)
            .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
        last_index = batch.n_tokens() - 1;
    }
    Ok((last_index, pos))
}

/// The role name llama.cpp templates expect.
fn role_name(role: PromptRole) -> &'static str {
    match role {
        PromptRole::System => "system",
        PromptRole::User => "user",
        PromptRole::Assistant => "assistant",
    }
}

/// The ChatML fallback layout.
fn chatml_prompt(prompt: &[PromptMessage]) -> String {
    let mut text = String::new();
    for message in prompt {
        text.push_str("<|im_start|>");
        text.push_str(role_name(message.role));
        text.push('\n');
        text.push_str(&message.text);
        text.push_str("<|im_end|>\n");
    }
    text.push_str("<|im_start|>assistant\n");
    text
}

/// Hands complete UTF-8 text to `on_token`, holding partial characters back.
fn flush_text(buffer: &mut Vec<u8>, on_token: &mut dyn FnMut(&str)) {
    loop {
        match std::str::from_utf8(buffer) {
            Ok(text) => {
                if !text.is_empty() {
                    on_token(text);
                }
                buffer.clear();
                return;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                if valid > 0 {
                    if let Ok(text) = std::str::from_utf8(&buffer[..valid]) {
                        on_token(text);
                    }
                    buffer.drain(..valid);
                    continue;
                }
                match error.error_len() {
                    // Bytes that can never become valid text: replace them so
                    // the rest of the reply is not lost.
                    Some(length) => {
                        buffer.drain(..length);
                        on_token("\u{FFFD}");
                    }
                    // Incomplete at the end of the buffer: wait for more.
                    None => return,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_prefix_stops_at_the_first_difference() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 9, 3]), 1);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[9]), 0);
        assert_eq!(common_prefix_len::<u8>(&[], &[1]), 0);
        assert_eq!(common_prefix_len::<u8>(&[], &[]), 0);
    }

    #[test]
    fn reusable_len_always_leaves_one_prompt_token() {
        // The prompt is exactly the cache: one token must still be decoded.
        assert_eq!(reusable_len(4, 4), 3);
        // The cache is a strict prefix of the prompt.
        assert_eq!(reusable_len(4, 10), 4);
        // Nothing in common.
        assert_eq!(reusable_len(0, 5), 0);
    }

    #[test]
    fn summarizer_context_rounds_to_256_and_checks_the_window() {
        assert_eq!(summarizer_context_size(10, 10, 4096), Some(256));
        assert_eq!(summarizer_context_size(0, 0, 256), Some(256));
        // 1000 + 100 + 8 = 1108, rounded up to 1280.
        assert_eq!(summarizer_context_size(1000, 100, 4096), Some(1280));
        assert_eq!(summarizer_context_size(1000, 100, 1280), Some(1280));
        assert_eq!(summarizer_context_size(1000, 100, 1279), None);
    }
}
