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

#[test]
#[ignore = "needs REGENT_TEST_MODEL pointing at a GGUF model"]
fn llama_chat_engine_saves_and_loads_its_prefix_cache() {
    let Ok(model_path) = std::env::var("REGENT_TEST_MODEL") else {
        println!("skipped: set REGENT_TEST_MODEL");
        return;
    };

    let config = EngineConfig {
        model_path: model_path.into(),
        gpu_layers: 0,
        context_size: 2048,
    };
    let system = PromptMessage {
        role: PromptRole::System,
        text: "You are a helpful assistant. Reply in one short sentence.".to_string(),
    };
    let prompt = vec![
        system.clone(),
        PromptMessage {
            role: PromptRole::User,
            text: "What is the capital of France?".to_string(),
        },
    ];
    let cancel = AtomicBool::new(false);

    let dir = tempfile::tempdir().expect("temp dir");
    let state = dir.path().join("prefix.state");

    // Engine A answers one turn, then saves the state for its fixed prefix.
    let mut engine = LlamaChatEngine::load(&config).expect("the first engine should load");
    let mut reply = String::new();
    engine
        .generate(&prompt, &cancel, &mut |piece| reply.push_str(piece))
        .expect("generation should succeed with a valid GGUF model");
    assert!(!reply.trim().is_empty(), "expected a non-empty reply");

    engine
        .save_prefix(std::slice::from_ref(&system), &state)
        .expect("saving the prefix should succeed");
    let state_len = std::fs::metadata(&state)
        .expect("the state file should exist")
        .len();
    assert!(state_len > 0, "expected a non-empty state file");

    // Engine B loads that state and reuses it for the same prompt.
    let mut fresh = LlamaChatEngine::load(&config).expect("the second engine should load");
    assert!(
        fresh
            .load_prefix(std::slice::from_ref(&system), &state)
            .expect("loading the prefix should succeed"),
        "expected the saved prefix to load"
    );

    let mut second_reply = String::new();
    let outcome = fresh
        .generate(&prompt, &cancel, &mut |piece| {
            second_reply.push_str(piece);
        })
        .expect("generation after loading should succeed");
    let Outcome::Finished(stats) = outcome else {
        panic!("expected a finished reply, got {outcome:?}");
    };
    println!("reused after load: {stats:?}");
    assert!(
        stats.reused_tokens > 0,
        "expected the loaded prefix to be reused, got {stats:?}"
    );

    // A prefix that differs from the saved one is never loaded.
    let other_system = PromptMessage {
        role: PromptRole::System,
        text: "You are a pirate. Reply in one short sentence.".to_string(),
    };
    assert!(
        !fresh
            .load_prefix(std::slice::from_ref(&other_system), &state)
            .expect("loading a different prefix should not fail"),
        "a different prefix must not load the saved state"
    );
}
