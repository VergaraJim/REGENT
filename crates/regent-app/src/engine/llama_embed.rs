//! The llama.cpp implementation of [`EmbeddingEngine`].
//!
//! The engine keeps one model and one context alive for its whole life and
//! embeds one text at a time. Every text is tokenised, truncated to the context
//! window and decoded in a single batch, because a non-causal embedding model
//! must see the whole input at once. The pooled vector is read from sequence 0
//! when the model has a pooling type, and averaged over the tokens otherwise.

use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use regent_core::chat::ChatEngineError;
use regent_core::memory::{EmbeddingConfig, EmbeddingEngine};

use super::llama::backend;
use super::llama_chat::thread_count;

/// The smallest context an embedding model is loaded with.
const MIN_CONTEXT: u32 = 256;

/// Plain-language messages for every failure.
const BACKEND_FAILED: &str =
    "The memory engine couldn't start on this PC. Restart Regent and try again.";
const MODEL_FAILED: &str =
    "The memory model couldn't be loaded. It may be damaged or not an embedding model.";
const CONTEXT_FAILED: &str =
    "The memory model couldn't start on this PC. Try a smaller context for it in Settings.";
const EMPTY_TEXT: &str = "There was nothing to embed.";
const DECODE_FAILED: &str = "The memory model stopped unexpectedly.";

/// A local embedding model served by llama.cpp.
pub struct LlamaEmbeddingEngine {
    /// Borrows `_model`; declared first so it is dropped before the model.
    context: LlamaContext<'static>,
    /// Owns the model; kept only so the context stays valid and is freed last.
    _model: Box<LlamaModel>,
    /// The context window that was actually loaded.
    context_size: usize,
    /// The length of every vector this engine produces.
    dimension: usize,
}

impl LlamaEmbeddingEngine {
    /// Loads the model and creates the single context used for every text.
    pub fn load(config: &EmbeddingConfig) -> Result<Self, ChatEngineError> {
        let backend = backend().map_err(|_| ChatEngineError(BACKEND_FAILED.to_string()))?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
        let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)
            .map_err(|_| ChatEngineError(MODEL_FAILED.to_string()))?;
        let model = Box::new(model);

        // At least 256 tokens, never more than the model was trained for.
        let mut context_size = config.context_size.max(MIN_CONTEXT);
        let train_context = model.n_ctx_train();
        if train_context > 0 {
            context_size = context_size.min(train_context);
        }
        let n_ctx = NonZeroU32::new(context_size)
            .ok_or_else(|| ChatEngineError(CONTEXT_FAILED.to_string()))?;

        let dimension = usize::try_from(model.n_embd()).unwrap_or(0);
        if dimension == 0 {
            return Err(ChatEngineError(MODEL_FAILED.to_string()));
        }

        // A non-causal model must see a whole input in one micro-batch, so the
        // batch sizes are as large as the context itself.
        let context_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(context_size)
            .with_n_ubatch(context_size)
            .with_n_threads(thread_count())
            .with_n_threads_batch(thread_count());

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
            dimension,
        })
    }

    /// Tokenises `text`, adding BOS only when the vocabulary wants it, and
    /// truncating to the loaded context window.
    fn tokenize(&self, text: &str) -> Vec<llama_cpp_2::token::LlamaToken> {
        let vocab = self._model.vocab();
        let mut tokens = vocab.tokenize(text.as_bytes(), false, true);
        let starts_with_bos = tokens.first().is_some_and(|token| *token == vocab.bos());
        if !starts_with_bos && vocab.should_add_bos() {
            tokens.insert(0, vocab.bos());
        }
        tokens.truncate(self.context_size);
        tokens
    }

    /// Embeds one text, returning a raw (un-normalised) vector.
    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>, ChatEngineError> {
        let tokens = self.tokenize(text);
        if tokens.is_empty() {
            return Err(ChatEngineError(EMPTY_TEXT.to_string()));
        }

        self.context.clear_kv_cache();

        let token_count = tokens.len();
        let mut batch = LlamaBatch::new(token_count, 1);
        for (position, &token) in tokens.iter().enumerate() {
            let position = i32::try_from(position).unwrap_or(i32::MAX);
            batch
                .add(token, position, &[0], true)
                .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
        }

        // Causal decoders embed with `decode`; encoder-only models need
        // `encode`, so a failed decode gets one more chance.
        if self.context.decode(&mut batch).is_err() {
            batch.clear();
            for (position, &token) in tokens.iter().enumerate() {
                let position = i32::try_from(position).unwrap_or(i32::MAX);
                batch
                    .add(token, position, &[0], true)
                    .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
            }
            self.context
                .encode(&mut batch)
                .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
        }

        let pooled = match self.context.embeddings_seq_ith(0) {
            Ok(vector) => vector.to_vec(),
            Err(_) => {
                // No pooling type: average the per-token vectors instead.
                let mut rows = Vec::with_capacity(token_count);
                for index in 0..token_count {
                    let index = i32::try_from(index).unwrap_or(i32::MAX);
                    let row = self
                        .context
                        .embeddings_ith(index)
                        .map_err(|_| ChatEngineError(DECODE_FAILED.to_string()))?;
                    rows.push(row.to_vec());
                }
                mean_pool(&rows)
            }
        };

        if pooled.len() != self.dimension {
            return Err(ChatEngineError(MODEL_FAILED.to_string()));
        }
        Ok(pooled)
    }
}

impl EmbeddingEngine for LlamaEmbeddingEngine {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, ChatEngineError> {
        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            vectors.push(self.embed_one(text)?);
        }
        Ok(vectors)
    }
}

/// Averages `rows` element by element.
///
/// Rows of a different length than the first are ignored; an empty input gives
/// an empty vector.
fn mean_pool(rows: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = rows.first() else {
        return Vec::new();
    };
    let mut sum = vec![0.0f32; first.len()];
    let mut count = 0usize;
    for row in rows {
        if row.len() != sum.len() {
            continue;
        }
        for (total, value) in sum.iter_mut().zip(row) {
            *total += value;
        }
        count += 1;
    }
    if count == 0 {
        return Vec::new();
    }
    let divisor = count as f32;
    sum.into_iter().map(|value| value / divisor).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_two_rows() {
        let rows = vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
        assert_eq!(mean_pool(&rows), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn mean_pool_leaves_a_single_row_unchanged() {
        let rows = vec![vec![0.5, -1.0]];
        assert_eq!(mean_pool(&rows), vec![0.5, -1.0]);
    }

    #[test]
    fn mean_pool_of_nothing_is_empty() {
        assert!(mean_pool(&[]).is_empty());
    }
}
