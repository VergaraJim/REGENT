//! The state every Tauri command shares.
//!
//! It is created once in `setup` and handed to commands through Tauri's managed
//! state. The estimator owns its own worker thread, so computing a forecast
//! never blocks the UI.

use std::sync::Mutex;

use regent_core::estimate::{Estimate, Estimator, FsInspector};
use regent_core::hardware::SystemProbe;
use regent_core::paths::AppPaths;
use regent_core::settings::{Settings, SettingsStore};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

/// The event emitted whenever a new memory forecast is ready.
pub const ESTIMATE_READY: &str = "estimate-ready";

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
    /// Every folder and file Regent uses.
    // TODO(settings UI): read by the commands that open folders and models.
    #[allow(dead_code)]
    pub paths: AppPaths,
}

impl AppState {
    /// Opens the settings and starts the estimator.
    ///
    /// Every finished forecast is emitted as [`ESTIMATE_READY`] on `app`.
    pub fn new(paths: AppPaths, app: AppHandle) -> Self {
        let store = SettingsStore::open(paths.settings_file.clone());
        let estimator = Estimator::spawn(
            std::sync::Arc::new(SystemProbe::new()),
            std::sync::Arc::new(FsInspector::new()),
            Box::new(move |generation, estimate| {
                let payload = EstimateReady {
                    generation,
                    estimate,
                };
                let _ = app.emit(ESTIMATE_READY, payload);
            }),
        );

        Self {
            store: Mutex::new(store),
            estimator,
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
