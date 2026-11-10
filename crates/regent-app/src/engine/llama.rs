//! Minimal CPU-only llama.cpp engine, used to prove that llama.cpp builds,
//! links and runs inside `regent-app`.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

/// Number of context tokens for the smoke generation.
const SMOKE_N_CTX: u32 = 512;

/// Errors from the llama.cpp inference path, in plain language.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// llama.cpp's global backend could not be initialised.
    #[error("could not initialise the llama.cpp backend: {0}")]
    BackendInit(String),

    /// The GGUF model file could not be loaded.
    #[error("could not load the model from {path}: {source}")]
    ModelLoad {
        /// Model path that was attempted.
        path: String,
        /// Underlying llama.cpp error.
        source: llama_cpp_2::LlamaModelLoadError,
    },

    /// An inference context (KV cache, thread pool) could not be created.
    #[error("could not create an inference context: {0}")]
    ContextLoad(#[from] llama_cpp_2::LlamaContextLoadError),

    /// The prompt could not be tokenised.
    #[error("could not tokenise the prompt: {0}")]
    DecodePrompt(llama_cpp_2::DecodeError),

    /// A generated token could not be converted back into text.
    #[error("could not turn generated tokens into text: {0}")]
    PieceToString(#[from] std::string::FromUtf8Error),

    /// A token could not be added to the decode batch.
    #[error("could not add a token to the batch: {0}")]
    BatchAdd(#[from] llama_cpp_2::llama_batch::BatchAddError),

    /// llama.cpp failed while decoding (running the forward pass).
    #[error("the model failed while processing: {0}")]
    Decode(#[from] llama_cpp_2::DecodeError),
}

/// Everything the smoke call reports back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmokeOutput {
    /// The generated text (not echoed prompt), decoded lossily-free pieces.
    pub text: String,
    /// Number of tokens the prompt tokenised into.
    pub prompt_tokens: usize,
    /// Number of tokens actually generated (may be fewer than requested if
    /// the model emits an end-of-generation token first).
    pub generated_tokens: usize,
}

/// Initialise llama.cpp's global backend exactly once per process.
///
/// Calling `llama_backend_init` twice makes llama.cpp abort its own global
/// state: always go through this function, never call `LlamaBackend::init`
/// directly more than once.
fn backend() -> Result<&'static LlamaBackend, EngineError> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

    let init = BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string()))
        .as_ref();
    match init {
        Ok(b) => Ok(b),
        Err(e) => Err(EngineError::BackendInit(e.clone())),
    }
}

/// Generate text greedily from a GGUF model, on the CPU, to prove the
/// engineered llama.cpp link works end to end.
///
/// - loads the model with zero GPU layers and default memory mapping;
/// - builds a 512-token context;
/// - threads: `max(1, available_parallelism - 1)`;
/// - tokens the prompt with the BOS token attached and decodes it;
/// - greedy-samples up to `max_tokens` tokens, stopping early on
///   end-of-generation.
///
/// # Errors
///
/// See [`EngineError`].
pub fn smoke_generate(
    model_path: &Path,
    prompt: &str,
    max_tokens: u32,
) -> Result<SmokeOutput, EngineError> {
    let backend = backend()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(backend, model_path, &model_params).map_err(|e| {
        EngineError::ModelLoad {
            path: model_path.display().to_string(),
            source: e,
        }
    })?;

    let n_ctx = NonZeroU32::new(SMOKE_N_CTX).expect("SMOKE_N_CTX is a hard-coded non-zero value");
    let threads = i32::try_from(
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .saturating_sub(1)
            .max(1),
    )
    .unwrap_or(1);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_threads(threads)
        .with_n_threads_batch(threads);
    let mut ctx = model.new_context(backend, ctx_params)?;

    // Tokenise with BOS + parse-special so the model sees a proper opening.
    let tokens = model.vocab().tokenize(prompt.as_bytes(), true, true);

    // Fill the prompt batch; only the final token computes logits.
    let prompt_len = tokens.len();
    let mut batch = LlamaBatch::new(prompt_len, 1);
    for (i, &token) in tokens.iter().enumerate() {
        batch
            .add(token, i as i32, &[0], i == prompt_len - 1)
            .map_err(EngineError::BatchAdd)?;
    }
    let last_logits_index = (prompt_len - 1) as i32;
    ctx.decode(&mut batch).map_err(EngineError::DecodePrompt)?;

    // Greedy sampling until end-of-generation or the token budget.
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let vocab = model.vocab();
    let mut generated: Vec<u8> = Vec::new();
    let mut generated_tokens = 0usize;
    for pos in (prompt_len as i32..).take(max_tokens as usize) {
        let token = sampler.sample(&ctx, last_logits_index);
        if vocab.is_eog(token) {
            break;
        }
        sampler.accept(token);
        generated_tokens += 1;

        generated.extend(vocab.token_to_piece(token, false, None));

        batch.clear();
        batch.add(token, pos, &[0], true)?;
        ctx.decode(&mut batch)?;
    }

    Ok(SmokeOutput {
        text: String::from_utf8(generated)?,
        prompt_tokens: prompt_len,
        generated_tokens,
    })
}
