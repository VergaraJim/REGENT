//! Opt-in smoke test for the llama.cpp chat engine.
//!
//! Run with a real, local GGUF model:
//! `REGENT_TEST_MODEL=C:\path\to\model.gguf cargo test -p regent-app --test
//! llama_chat_smoke -- --ignored`
#![cfg(feature = "llama")]

use std::sync::atomic::AtomicBool;

use regent_app_lib::engine::LlamaChatEngine;
use regent_core::chat::{
    build_summary_prompt, ChatEngine, ChatMessage, EngineConfig, Outcome, PromptMessage,
    PromptRole, Role,
};

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

#[test]
#[ignore = "needs REGENT_TEST_MODEL pointing at a GGUF model"]
fn llama_chat_engine_reuses_its_cache_and_summarizes() {
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
    let cancel = AtomicBool::new(false);

    let mut prompt = vec![
        PromptMessage {
            role: PromptRole::System,
            text: "You are a helpful assistant. Reply in one short sentence.".to_string(),
        },
        PromptMessage {
            role: PromptRole::User,
            text: "What is the capital of France?".to_string(),
        },
    ];

    let mut reply = String::new();
    engine
        .generate(&prompt, &cancel, &mut |piece| reply.push_str(piece))
        .expect("the first generation should succeed with a valid GGUF model");
    assert!(!reply.trim().is_empty(), "expected a non-empty first reply");

    // The second turn repeats the whole first prompt, so most of it is served
    // from the cache instead of being decoded again.
    prompt.push(PromptMessage {
        role: PromptRole::Assistant,
        text: reply.clone(),
    });
    prompt.push(PromptMessage {
        role: PromptRole::User,
        text: "And what is the capital of Italy?".to_string(),
    });

    let mut second_reply = String::new();
    let outcome = engine
        .generate(&prompt, &cancel, &mut |piece| {
            second_reply.push_str(piece);
        })
        .expect("the second generation should succeed");
    let Outcome::Finished(stats) = outcome else {
        panic!("expected a finished second reply, got {outcome:?}");
    };
    println!("reused: {stats:?}");
    assert!(
        stats.reused_tokens > 0,
        "expected some prompt tokens to be reused, got {stats:?}"
    );
    assert!(
        stats.processed_tokens < stats.prompt_tokens,
        "expected fewer processed than prompt tokens, got {stats:?}"
    );

    let fold = vec![
        ChatMessage {
            id: 1,
            role: Role::User,
            text: "What is the capital of France?".to_string(),
            from_voice: false,
            created_at: 0,
        },
        ChatMessage {
            id: 2,
            role: Role::Assistant,
            text: reply,
            from_voice: false,
            created_at: 0,
        },
    ];
    let summary_prompt = build_summary_prompt(None, &fold);
    let summary = engine
        .summarize(&summary_prompt, 64, &cancel)
        .expect("summarizing should succeed");
    let text = summary.expect("summary should not be cancelled");
    println!("summary: {text}");
    assert!(!text.trim().is_empty(), "expected a non-blank summary");
}
