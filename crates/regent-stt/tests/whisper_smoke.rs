//! Opt-in smoke test for the whisper.cpp backend.
//!
//! Run with:
//! `REGENT_TEST_STT_MODEL=C:\path\to\ggml-base.en.bin cargo test -p regent-stt --test whisper_smoke -- --nocapture`
#![cfg(feature = "whisper")]

use regent_stt::service::Transcriber;

#[test]
fn transcribes_one_second_of_silence() {
    let model = match std::env::var("REGENT_TEST_STT_MODEL") {
        Ok(model) if !model.is_empty() => model,
        _ => {
            println!("skipped: REGENT_TEST_STT_MODEL is not set");
            return;
        }
    };

    let mut transcriber = regent_stt::whisper::WhisperTranscriber::load(&model, "auto", 1)
        .expect("model should load");

    let samples = vec![0.0f32; 16_000];
    let result = transcriber.transcribe(&samples);
    println!("silence transcript: {result:?}");
    assert!(result.is_ok(), "transcription failed: {result:?}");
}
