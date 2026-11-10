//! The state every Tauri command shares.
//!
//! It is created once in `setup` and handed to commands through Tauri's managed
//! state. The estimator and the chat service each own a worker thread, so the
//! UI never blocks on a forecast or a reply.

use std::sync::Mutex;

use regent_core::chat::{ChatService, EngineFactory};
use regent_core::estimate::{Estimate, Estimator, FsInspector};
use regent_core::hardware::SystemProbe;
use regent_core::paths::AppPaths;
use regent_core::settings::{Settings, SettingsStore};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

/// The event emitted whenever a new memory forecast is ready.
pub const ESTIMATE_READY: &str = "estimate-ready";

/// The event emitted whenever the chat service changes state.
pub const CHAT_STATE: &str = "chat-state";

/// The error reported by a build without the llama feature.
#[cfg(not(feature = "llama"))]
const NO_ENGINE: &str = "This build of Regent has no text engine.";

/// The payload of [`ESTIMATE_READY`].
#[derive(Debug, Clone, Serialize)]
pub struct EstimateReady {
    /// The generation this forecast answers; compare it against the newest
    /// request so stale results can be ignored.
    pub generation: u64,
    /// The forecast itself.
    pub estimate: Estimate,
}

/// Everything the commands need, for as long as the app runs.
pub struct AppState {
    /// The settings and the file they are saved to.
    pub store: Mutex<SettingsStore>,
    /// The worker that computes memory forecasts.
    pub estimator: Estimator,
    /// The worker that owns the text model and the chat history.
    pub chat: ChatService,
    /// Every folder and file Regent uses.
    // TODO(settings UI): read by the commands that open folders and models.
    #[allow(dead_code)]
    pub paths: AppPaths,
}

impl AppState {
    /// Opens the settings and starts the estimator and chat workers.
    ///
    /// Every finished forecast is emitted as [`ESTIMATE_READY`] on `app`, and
    /// every chat state change as [`CHAT_STATE`].
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
        let chat = ChatService::spawn(
            paths.database_file.clone(),
            engine_factory(),
            Box::new(move |status| {
                let _ = app.emit(CHAT_STATE, status);
            }),
        );

        Self {
            store: Mutex::new(store),
            estimator,
            chat,
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
