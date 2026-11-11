//! The llama.cpp implementation of [`ChatEngine`].
//!
//! The engine keeps one model and one context alive for its whole life. Every
//! turn renders the prompt with the model's own chat template (falling back to
//! ChatML), clears the KV cache and decodes the whole prompt again: no prefix
//! cache is reused in this stage.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use regent_core::chat::{
    ChatEngine, ChatEngineError, EngineConfig, GenerationStats, Outcome, PromptMessage, PromptRole,
    REPLY_RESERVE_TOKENS,
};
use regent_core::context::{clamp_requested, context_limits, MIN_CONTEXT};

use super::llama::backend;

/// The fixed seed keeps a replay of the same prompt reproducible.
const SAMPLE_SEED: u32 = 0x5EED_1234;

/// Sampling: keep the 40 best tokens, then the smallest set covering 95% of
/// the probability, then apply the temperature, then draw.
const TOP_K: i32 = 40;
const TOP_P: f32 = 0.95;
const TEMPERATURE: f32 = 0.7;

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
        let threads = i32::try_from(
            std::thread::available_parallelism()
                .map_or(1, |count| count.get())
                .saturating_sub(1)
                .max(1),
        )
        .unwrap_or(1);
        let context_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_threads(threads)
            .with_n_threads_batch(threads);

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

    fn generate(
        &mut self,
        prompt: &[PromptMessage],
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Outcome, ChatEngineError> {
        let rendered = self.render_prompt(prompt);
        self.context.clear_kv_cache();

        let tokens = self.tokenize_prompt(&rendered);
        let prompt_len = tokens.len();
        if prompt_len == 0 {
            return Err(ChatEngineError(EMPTY_PROMPT.to_string()));
        }
        if prompt_len >= self.context_size {
            return Err(ChatEngineError(TOO_LONG.to_string()));
        }

        // Decode the prompt in chunks no larger than the context's batch size;
        // only the very last token computes logits.
        let batch_size = usize::try_from(self.context.n_batch()).unwrap_or(1).max(1);
        let mut batch = LlamaBatch::new(batch_size.min(prompt_len), 1);
        let mut pos = 0i32;
        for (chunk_index, chunk) in tokens.chunks(batch_size).enumerate() {
            let chunk_start = chunk_index * batch_size;
            batch.clear();
            for (offset, &token) in chunk.iter().enumerate() {
                let is_last = chunk_start + offset + 1 == prompt_len;
                batch
                    .add(token, pos, &[0], is_last)
                    .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
                pos += 1;
            }
            self.context
                .decode(&mut batch)
                .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
        }
        let mut last_index = batch.n_tokens() - 1;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(TOP_K),
            LlamaSampler::top_p(TOP_P, 1),
            LlamaSampler::temp(TEMPERATURE),
            LlamaSampler::dist(SAMPLE_SEED),
        ]);

        let max_new = (self.context_size - prompt_len).min(REPLY_RESERVE_TOKENS);
        let started = Instant::now();
        let mut generated_tokens = 0usize;
        let mut pending: Vec<u8> = Vec::new();
        let mut cancelled = false;

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
            batch
                .add(token, pos, &[0], true)
                .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
            self.context
                .decode(&mut batch)
                .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
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
            generated_tokens,
            tokens_per_second,
        };

        if cancelled {
            Ok(Outcome::Cancelled(stats))
        } else {
            Ok(Outcome::Finished(stats))
        }
    }
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
