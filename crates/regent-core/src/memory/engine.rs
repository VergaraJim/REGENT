//! The embedding model abstraction: what memory indexing needs from a model,
//! plus a deterministic [`FakeEmbedder`] for tests.
//!
//! Nothing here touches llama.cpp. The real embedder is adapted to
//! [`EmbeddingEngine`] in a later stage; everything above the trait is tested
//! with the fake.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::chat::ChatEngineError;
use crate::hash::fnv1a64;

/// How to load an embedding model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingConfig {
    /// Path to the model file.
    pub model_path: PathBuf,
    /// Layers offloaded to the GPU.
    pub gpu_layers: u32,
    /// Context window in tokens.
    pub context_size: u32,
    /// Instruction prefix used when embedding stored documents.
    pub document_prefix: Option<String>,
    /// Instruction prefix used when embedding a search query.
    pub query_prefix: Option<String>,
}

/// The local embedding model as the memory indexer sees it.
pub trait EmbeddingEngine: Send {
    /// The length of every vector this engine produces.
    fn dimension(&self) -> usize;

    /// Embeds `texts`, returning one raw vector per text in the same order.
    ///
    /// The vectors are not normalised; the caller normalises them.
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, ChatEngineError>;
}

/// Builds an embedding engine from a configuration; the Tauri layer supplies
/// the real one, tests supply a fake.
pub type EmbeddingFactory = Box<
    dyn Fn(&EmbeddingConfig) -> Result<Box<dyn EmbeddingEngine>, ChatEngineError> + Send + Sync,
>;

/// The dimension [`FakeEmbedder`] always produces.
const FAKE_DIMENSION: usize = 8;

/// A deterministic embedder for tests.
///
/// Each whitespace-separated lowercase word adds 1.0 at index
/// `fnv1a64(word) % 8`. Text with no words gives `[1, 0, 0, …]`.
pub struct FakeEmbedder {
    dimension: usize,
    failures: u32,
    log: Option<Arc<Mutex<Vec<String>>>>,
}

impl FakeEmbedder {
    /// An embedder that succeeds immediately.
    pub fn new() -> FakeEmbedder {
        FakeEmbedder {
            dimension: FAKE_DIMENSION,
            failures: 0,
            log: None,
        }
    }

    /// Makes the next `n` calls to [`EmbeddingEngine::embed`] return an error.
    pub fn with_failures(mut self, n: u32) -> Self {
        self.failures = n;
        self
    }

    /// Records every text passed to [`EmbeddingEngine::embed`] into `log`.
    pub fn with_log(mut self, log: Arc<Mutex<Vec<String>>>) -> Self {
        self.log = Some(log);
        self
    }
}

impl Default for FakeEmbedder {
    fn default() -> Self {
        FakeEmbedder::new()
    }
}

impl EmbeddingEngine for FakeEmbedder {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, ChatEngineError> {
        if self.failures > 0 {
            self.failures -= 1;
            return Err(ChatEngineError("The memory model failed.".to_string()));
        }

        if let Some(log) = &self.log {
            let mut log = log.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            log.extend(texts.iter().cloned());
        }

        Ok(texts
            .iter()
            .map(|text| embed_one(text, self.dimension))
            .collect())
    }
}

/// Embeds one text with the fake's hashing scheme.
fn embed_one(text: &str, dimension: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    let mut words = 0;
    for word in text.split_whitespace() {
        let word = word.to_lowercase();
        let index = (fnv1a64(word.as_bytes()) % dimension as u64) as usize;
        vector[index] += 1.0;
        words += 1;
    }
    if words == 0 {
        vector[0] = 1.0;
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fake_has_dimension_eight() {
        assert_eq!(FakeEmbedder::new().dimension(), 8);
    }

    #[test]
    fn words_accumulate_and_text_with_no_words_is_a_unit_basis_vector() {
        let mut embedder = FakeEmbedder::new();
        let vectors = embedder
            .embed(&["hello hello".to_string(), "   ".to_string()])
            .expect("embed");

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0].len(), 8);
        assert_eq!(vectors[0].iter().sum::<f32>(), 2.0);
        assert_eq!(vectors[1], vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn lowercase_words_hash_to_the_same_slot() {
        let mut embedder = FakeEmbedder::new();
        let upper = embedder.embed(&["HELLO".to_string()]).expect("embed");
        let lower = embedder.embed(&["hello".to_string()]).expect("embed");
        assert_eq!(upper, lower);
    }

    #[test]
    fn failures_count_down_then_stop() {
        let mut embedder = FakeEmbedder::new().with_failures(2);
        assert!(embedder.embed(&["one".to_string()]).is_err());
        assert!(embedder.embed(&["two".to_string()]).is_err());
        assert!(embedder.embed(&["three".to_string()]).is_ok());
    }

    #[test]
    fn the_log_records_every_text() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut embedder = FakeEmbedder::new().with_log(Arc::clone(&log));
        embedder
            .embed(&["a".to_string(), "b".to_string()])
            .expect("embed");
        assert_eq!(
            log.lock().unwrap().clone(),
            vec!["a".to_string(), "b".to_string()]
        );
    }
}
