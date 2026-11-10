//! whisper.cpp backend, behind the default `whisper` feature.

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::service::Transcriber;

/// A loaded whisper.cpp model. Each request creates a fresh state, so requests
/// cannot leak context into each other.
pub struct WhisperTranscriber {
    context: WhisperContext,
    language: Option<String>,
    threads: i32,
}

impl WhisperTranscriber {
    /// Load a model file. `language` is `"auto"` or a whisper language code.
    pub fn load(model: impl AsRef<Path>, language: &str, threads: u32) -> Result<Self, String> {
        // whisper-rs without a log backend routes whisper.cpp/ggml logs into
        // hooks that discard them. Without this, whisper.cpp writes its logs
        // to stdout, which would corrupt the JSON-line protocol.
        whisper_rs::install_logging_hooks();

        let context =
            WhisperContext::new_with_params(model.as_ref(), WhisperContextParameters::default())
                .map_err(|error| format!("failed to load whisper model: {error}"))?;

        let language = match language {
            "auto" | "" => None,
            code => Some(code.to_string()),
        };

        Ok(Self {
            context,
            language,
            threads: threads.max(1) as i32,
        })
    }
}

impl Transcriber for WhisperTranscriber {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        let mut state = self
            .context
            .create_state()
            .map_err(|error| format!("failed to create whisper state: {error}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads);
        params.set_language(self.language.as_deref());
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);

        state
            .full(params, samples)
            .map_err(|error| format!("transcription failed: {error}"))?;

        let mut text = String::new();
        for index in 0..state.full_n_segments() {
            let segment = state
                .get_segment(index)
                .ok_or_else(|| format!("whisper returned segment {index} out of bounds"))?;
            let part = segment
                .to_str_lossy()
                .map_err(|error| format!("failed to read segment {index}: {error}"))?;
            text.push_str(&part);
        }

        Ok(text)
    }
}
