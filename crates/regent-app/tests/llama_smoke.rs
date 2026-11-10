//! Opt-in smoke test for the llama.cpp engine.
//!
//! Run with a real, local GGUF model:
//! `REGENT_TEST_TEXT_MODEL=C:\path\to\model.gguf cargo test -p regent-app --test llama_smoke`
#![cfg(feature = "llama")]

use std::path::Path;

use regent_app_lib::engine::smoke_generate;

#[test]
fn smoke_generate_produces_tokens() {
    let Ok(model_path) = std::env::var("REGENT_TEST_TEXT_MODEL") else {
        println!("skipped: set REGENT_TEST_TEXT_MODEL");
        return;
    };

    let output = smoke_generate(Path::new(&model_path), "Hello", 8)
        .expect("smoke generation should succeed with a valid GGUF model");

    println!("generated text: {}", output.text);
    println!(
        "prompt tokens: {}, generated tokens: {}",
        output.prompt_tokens, output.generated_tokens
    );
    assert!(output.generated_tokens > 0);
}
