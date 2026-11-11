//! The state every Tauri command shares.
//!
//! It is created once in `setup` and handed to commands through Tauri's managed
//! state. The estimator and the chat service each own a worker thread, so the
//! UI never blocks on a forecast or a reply. The voice services own their own
//! threads too, and report everything they do through events.

use std::path::PathBuf;
use std::sync::Mutex;

#[cfg(not(feature = "llama"))]
use regent_core::chat::ChatEngineError;
use regent_core::chat::{ChatService, ChatServiceDeps, EngineFactory};
use regent_core::estimate::{Estimate, Estimator, FsInspector};
use regent_core::hardware::SystemProbe;
use regent_core::memory::EmbeddingFactory;
use regent_core::paths::AppPaths;
use regent_core::settings::{Settings, SettingsStore};
use regent_core::voice::{
    Speaker, SpeakerDeps, SpeakerEvent, VoiceEvent, VoiceInput, VoiceInputDeps,
};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::audio::{audio_out_factory, mic_factory};
use crate::stt::{resolve_helper, stt_factory};
use crate::tts::synth_factory;

/// The event emitted whenever a new memory forecast is ready.
pub const ESTIMATE_READY: &str = "estimate-ready";

/// The event emitted whenever the chat service changes state.
pub const CHAT_STATE: &str = "chat-state";

/// The event emitted whenever the memory indexer changes state.
pub const MEMORY_STATUS: &str = "memory-status";

/// The event emitted whenever voice input changes state.
pub const VOICE_STATUS: &str = "voice-status";

/// The event emitted with the microphone level while speech is in progress.
pub const MIC_LEVEL: &str = "mic-level";

/// The event emitted when a spoken utterance has been transcribed.
pub const VOICE_TRANSCRIPT: &str = "voice-transcript";

/// The event emitted when Regent starts or stops speaking.
pub const SPEAKING: &str = "speaking";

/// The event emitted when the speaker cannot play something.
pub const VOICE_OUTPUT_ERROR: &str = "voice-output-error";

/// The error reported by a build without the llama feature.
#[cfg(not(feature = "llama"))]
const NO_ENGINE: &str = "This build of Regent has no text engine.";

/// The placeholder memory-engine error for a build without the llama feature.
#[cfg(not(feature = "llama"))]
const NO_MEMORY_ENGINE: &str = "This build of Regent has no memory engine.";

/// The payload of [`ESTIMATE_READY`].
#[derive(Debug, Clone, Serialize)]
pub struct EstimateReady {
    /// The generation this forecast answers; compare it against the newest
    /// request so stale results can be ignored.
    pub generation: u64,
    /// The forecast itself.
    pub estimate: Estimate,
}

/// The payload of [`MIC_LEVEL`].
#[derive(Debug, Clone, Serialize)]
pub struct MicLevel {
    /// The microphone level, from `0.0` to `1.0`.
    pub level: f32,
}

/// The payload of [`VOICE_TRANSCRIPT`].
#[derive(Debug, Clone, Serialize)]
pub struct VoiceTranscript {
    /// The cleaned transcript.
    pub text: String,
}

/// The payload of [`SPEAKING`].
#[derive(Debug, Clone, Serialize)]
pub struct SpeakingPayload {
    /// Whether Regent is speaking now.
    pub speaking: bool,
}

/// The payload of [`VOICE_OUTPUT_ERROR`].
#[derive(Debug, Clone, Serialize)]
pub struct VoiceOutputError {
    /// The plain-language error.
    pub message: String,
}

/// Everything the commands need, for as long as the app runs.
pub struct AppState {
    /// The settings and the file they are saved to.
    pub store: Mutex<SettingsStore>,
    /// The worker that computes memory forecasts.
    pub estimator: Estimator,
    /// The worker that owns the text model and the chat history.
    pub chat: ChatService,
    /// The always-listening voice input service.
    pub voice: VoiceInput,
    /// The service that speaks replies aloud.
    pub speaker: Speaker,
    /// Where the app's bundled resources live, when the platform reports one.
    #[allow(dead_code)]
    pub resource_dir: Option<PathBuf>,
    /// Every folder and file Regent uses.
    // TODO(settings UI): read by the commands that open folders and models.
    #[allow(dead_code)]
    pub paths: AppPaths,
}

impl AppState {
    /// Opens the settings and starts the estimator, chat and voice workers.
    ///
    /// Every finished forecast is emitted as [`ESTIMATE_READY`] on `app`, every
    /// chat state change as [`CHAT_STATE`], and every voice change through its
    /// own event.
    pub fn new(paths: AppPaths, app: AppHandle) -> Self {
        let store = SettingsStore::open(paths.settings_file.clone());
        let estimate_app = app.clone();
        let estimator = Estimator::spawn(
            std::sync::Arc::new(SystemProbe::new()),
            std::sync::Arc::new(FsInspector::new()),
            Box::new(move |generation, estimate| {
                let payload = EstimateReady {
                    generation,
                    estimate,
                };
                let _ = estimate_app.emit(ESTIMATE_READY, payload);
            }),
        );
        let chat_app = app.clone();
        let memory_app = app.clone();
        let chat = ChatService::spawn(ChatServiceDeps {
            db_path: paths.database_file.clone(),
            prefix_cache_dir: paths.prefix_cache_dir.clone(),
            engine_factory: engine_factory(),
            embedding_factory: embedding_factory(),
            status_sink: Box::new(move |status| {
                let _ = chat_app.emit(CHAT_STATE, status);
            }),
            memory_sink: Box::new(move |status| {
                let _ = memory_app.emit(MEMORY_STATUS, status);
            }),
        });

        let resource_dir = app.path().resource_dir().ok();

        let voice_app = app.clone();
        let voice = VoiceInput::spawn(VoiceInputDeps {
            mic_factory: mic_factory(),
            stt_factory: stt_factory(resolve_helper(), paths.log_dir.join("regent-stt.log")),
            sink: Box::new(move |event| match event {
                VoiceEvent::Status(status) => {
                    let _ = voice_app.emit(VOICE_STATUS, status);
                }
                VoiceEvent::Level(level) => {
                    let _ = voice_app.emit(MIC_LEVEL, MicLevel { level });
                }
                VoiceEvent::Transcript(text) => {
                    let _ = voice_app.emit(VOICE_TRANSCRIPT, VoiceTranscript { text });
                }
            }),
        });

        let speaker_app = app.clone();
        let speaker = Speaker::spawn(SpeakerDeps {
            synth_factory: synth_factory(resource_dir.clone()),
            audio_factory: audio_out_factory(),
            sink: Box::new(move |event| match event {
                SpeakerEvent::Speaking(speaking) => {
                    let _ = speaker_app.emit(SPEAKING, SpeakingPayload { speaking });
                    let Some(state) = speaker_app.try_state::<AppState>() else {
                        return;
                    };
                    if speaking {
                        if state.settings().voice.pause_mic_while_speaking {
                            state.voice.set_paused(true);
                        }
                    } else {
                        // Always resume on the falling edge: turning the setting
                        // off mid-speech must not leave the mic paused.
                        state.voice.set_paused(false);
                    }
                }
                SpeakerEvent::Error(message) => {
                    let _ = speaker_app.emit(VOICE_OUTPUT_ERROR, VoiceOutputError { message });
                }
            }),
        });

        Self {
            store: Mutex::new(store),
            estimator,
            chat,
            voice,
            speaker,
            resource_dir,
            paths,
        }
    }

    /// A copy of the settings as they are now.
    pub fn settings(&self) -> Settings {
        self.store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .settings()
            .clone()
    }
}

/// Builds the engine factory for this build.
fn engine_factory() -> EngineFactory {
    #[cfg(feature = "llama")]
    {
        Box::new(|config| {
            crate::engine::LlamaChatEngine::load(config)
                .map(|engine| Box::new(engine) as Box<dyn regent_core::chat::ChatEngine>)
        })
    }
    #[cfg(not(feature = "llama"))]
    {
        Box::new(|_config| Err(regent_core::chat::ChatEngineError(NO_ENGINE.to_string())))
    }
}

/// Builds the embedding factory for this build.
fn embedding_factory() -> EmbeddingFactory {
    #[cfg(feature = "llama")]
    {
        Box::new(|config| {
            crate::engine::LlamaEmbeddingEngine::load(config)
                .map(|engine| Box::new(engine) as Box<dyn regent_core::memory::EmbeddingEngine>)
        })
    }
    #[cfg(not(feature = "llama"))]
    {
        Box::new(|_config| Err(ChatEngineError(NO_MEMORY_ENGINE.to_string())))
    }
}
