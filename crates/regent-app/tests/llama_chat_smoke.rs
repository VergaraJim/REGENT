//! Opt-in smoke test for the llama.cpp chat engine.
//!
//! Run with a real, local GGUF model:
//! `REGENT_TEST_MODEL=C:\path\to\model.gguf cargo test -p regent-app --test
//! llama_chat_smoke -- --ignored`
#![cfg(feature = "llama")]

use std::sync::atomic::AtomicBool;

use regent_app_lib::engine::LlamaChatEngine;
use regent_core::chat::{ChatEngine, EngineConfig, PromptMessage, PromptRole};

#[test]
#[ignore = "needs REGENT_TEST_MODEL pointing at a GGUF model"]
fn llama_chat_engine_generates_a_reply() {
    let Ok(model_path) = std::env::var("REGENT_TEST_MODEL") else {
        println!("skipped: set REGENT_TEST_MODEL");
        return;
    };

    let config = EngineConfig {
        model_path: model_path.into(),
        gpu_layers: 0,
        context_size: 2048,
    };
    let mut engine = LlamaChatEngine::load(&config).expect("the chat engine should load");

    let prompt = vec![PromptMessage {
        role: PromptRole::User,
        text: "Say hello in one short sentence.".to_string(),
    }];
    let cancel = AtomicBool::new(false);
    let mut reply = String::new();
    let outcome = engine
        .generate(&prompt, &cancel, &mut |piece| reply.push_str(piece))
        .expect("generation should succeed with a valid GGUF model");

    println!("reply: {reply}");
    println!("outcome: {outcome:?}");
    assert!(
        !reply.trim().is_empty(),
        "expected a non-empty reply, got {outcome:?}"
    );
}
