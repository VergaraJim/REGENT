//! Opt-in smoke test for the llama.cpp embedding engine.
//!
//! Run with a real, local embedding GGUF model:
//! `REGENT_TEST_EMBED_MODEL=C:\path\to\embedding.gguf cargo test -p regent-app
//! --test llama_embed_smoke -- --ignored`
#![cfg(feature = "llama")]

use regent_app_lib::engine::LlamaEmbeddingEngine;
use regent_core::memory::{dot, normalize, EmbeddingConfig, EmbeddingEngine};

#[test]
#[ignore = "needs REGENT_TEST_EMBED_MODEL pointing at an embedding GGUF model"]
fn llama_embed_engine_puts_related_texts_closer_together() {
    let Ok(model_path) = std::env::var("REGENT_TEST_EMBED_MODEL") else {
        println!("skipped: set REGENT_TEST_EMBED_MODEL");
        return;
    };

    let config = EmbeddingConfig {
        model_path: model_path.into(),
        gpu_layers: 0,
        context_size: 512,
        document_prefix: None,
        query_prefix: None,
    };
    let mut engine = LlamaEmbeddingEngine::load(&config).expect("the embedding engine should load");
    let dimension = engine.dimension();

    let texts = vec![
        "The cat sat on the mat.".to_string(),
        "A kitten is resting on the rug.".to_string(),
        "Quarterly tax filing deadline".to_string(),
    ];
    let mut vectors = engine
        .embed(&texts)
        .expect("embedding should succeed with a valid GGUF model");

    for vector in &mut vectors {
        assert_eq!(
            vector.len(),
            dimension,
            "every vector should have the dimension"
        );
        assert!(normalize(vector), "every vector should normalise");
    }

    let related = dot(&vectors[0], &vectors[1]);
    let unrelated = dot(&vectors[0], &vectors[2]);
    println!("related: {related}, unrelated: {unrelated}");
    assert!(
        related > unrelated,
        "the two cat sentences should be closer than the tax one: {related} <= {unrelated}"
    );
}
