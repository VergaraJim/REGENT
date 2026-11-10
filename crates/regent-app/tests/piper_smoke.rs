//! Opt-in smoke test for the bundled Piper engine.
//!
//! Needs a Piper voice on disk and a fetched Piper folder:
//! `REGENT_TEST_PIPER_VOICE=C:\path\to\en_US-lessac-low.onnx cargo test -p regent-app --no-default-features --test piper_smoke -- --nocapture`
//!
//! `REGENT_TEST_PIPER_DIR` overrides the Piper folder; it defaults to
//! `crates/regent-app/resources/piper`.

use std::path::PathBuf;

use regent_app_lib::tts;

#[test]
fn piper_speaks() {
    let Some(voice) = std::env::var_os("REGENT_TEST_PIPER_VOICE") else {
        println!("skipped: set REGENT_TEST_PIPER_VOICE to a Piper .onnx voice");
        return;
    };
    let voice = PathBuf::from(voice);

    let piper_dir = std::env::var_os("REGENT_TEST_PIPER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("resources")
                .join("piper")
        });

    // In debug builds `resolve_piper` honours this override.
    std::env::set_var("REGENT_PIPER_DIR", &piper_dir);

    let resource_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources");
    let Ok(paths) = tts::resolve_piper(&resource_dir) else {
        println!(
            "skipped: Piper is not fetched in {} (run `npm run fetch:piper`)",
            piper_dir.display()
        );
        return;
    };

    let samples =
        tts::synthesize_once(&paths, &voice, "Hello from Regent.").expect("synthesis should work");

    println!("synthesized {} samples", samples.len());
    assert!(!samples.is_empty(), "Piper should return audio samples");
}
