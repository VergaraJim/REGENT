//! Opt-in smoke test for the llama.cpp vision engine.
//!
//! Run with a real vision GGUF and its mmproj projector:
//! `REGENT_TEST_VISION_MODEL=C:\path\to\vision.gguf
//! REGENT_TEST_VISION_PROJECTOR=C:\path\to\mmproj.gguf cargo test -p regent-app
//! --test llama_vision_smoke -- --ignored`
#![cfg(feature = "llama")]

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use regent_app_lib::engine::LlamaVisionEngine;
use regent_core::vision::{ScreenImage, VisionConfig, VisionEngine};

#[test]
#[ignore = "needs REGENT_TEST_VISION_MODEL and REGENT_TEST_VISION_PROJECTOR pointing at a vision GGUF and its projector"]
fn llama_vision_engine_reads_a_solid_red_image() {
    let Ok(model_path) = std::env::var("REGENT_TEST_VISION_MODEL") else {
        println!("skipped: set REGENT_TEST_VISION_MODEL");
        return;
    };
    let Ok(projector_path) = std::env::var("REGENT_TEST_VISION_PROJECTOR") else {
        println!("skipped: set REGENT_TEST_VISION_PROJECTOR");
        return;
    };

    let config = VisionConfig {
        model_path: model_path.into(),
        projector_path: projector_path.into(),
        gpu_layers: 0,
        max_image_edge: 64,
        keep_alive: Duration::from_secs(60),
    };
    let mut engine = LlamaVisionEngine::load(&config).expect("the vision engine should load");

    let side = 64u32;
    let mut rgb = Vec::with_capacity((side * side * 3) as usize);
    for _ in 0..side * side {
        rgb.extend_from_slice(&[255, 0, 0]);
    }
    let image = ScreenImage::new(side, side, rgb).expect("the image should be valid");

    let answer = engine
        .answer(
            &image,
            "What colour fills this image?",
            64,
            &AtomicBool::new(false),
        )
        .expect("answering should succeed with a valid model and projector");

    println!("answer: {answer}");
    assert!(!answer.trim().is_empty(), "the answer should not be empty");
}
