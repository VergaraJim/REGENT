//! Tauri IPC commands.
//!
//! Every command must be `async`. Slow work must leave the main thread via
//! `tauri::async_runtime::spawn_blocking` so the UI never stalls.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use regent_core::chat::{ChatEvent, ChatMessage, ChatStatus, EngineConfig, HISTORY_READ_CAP};
use regent_core::estimate::SlotId;
use regent_core::settings::{SaveError, Settings, SettingsStore};
use serde::Serialize;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;

use crate::state::AppState;

/// Where the only window lives.
const MAIN_WINDOW: &str = "main";

/// The warning shown when a Piper voice has no sidecar config beside it.
const VOICE_CONFIG_WARNING: &str = "This voice needs its .onnx.json file in the same folder.";

/// The error when a background chat task cannot be awaited.
const CHAT_TASK_FAILED: &str = "Regent's chat service stopped unexpectedly. Restart Regent.";

#[derive(Serialize)]
pub struct AppInfo {
    name: &'static str,
    version: &'static str,
}

#[tauri::command]
pub async fn app_info() -> AppInfo {
    AppInfo {
        name: "Regent",
        version: env!("CARGO_PKG_VERSION"),
    }
}

/// A command failure, in words the user can read.
#[derive(Debug, Serialize)]
pub struct CommandError {
    message: String,
}

impl CommandError {
    /// Builds an error from a user-facing message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// The settings plus the state of the file they were read from.
#[derive(Debug, Clone, Serialize)]
pub struct SettingsView {
    /// The current settings.
    pub settings: Settings,
    /// Why an existing settings file couldn't be read, when it couldn't.
    pub load_error: Option<String>,
    /// Whether saving is refused until the user resets the file.
    pub saving_blocked: bool,
    /// Where the settings file lives.
    pub settings_path: String,
}

impl SettingsView {
    fn from_store(store: &SettingsStore) -> Self {
        Self {
            settings: store.settings().clone(),
            load_error: store.load_error().map(str::to_owned),
            saving_blocked: store.saving_blocked(),
            settings_path: store.path().display().to_string(),
        }
    }
}

/// A model file the user chose in the native dialog.
#[derive(Debug, Clone, Serialize)]
pub struct PickedFile {
    /// The absolute path, as a string the settings can store.
    pub path: String,
    /// Something about the file the user should know.
    pub warning: Option<String>,
}

/// Returns the settings, whether or not they could be loaded.
#[tauri::command]
pub async fn get_settings(app: AppHandle) -> SettingsView {
    let state = app.state::<AppState>();
    let store = lock_store(&state.store);
    SettingsView::from_store(&store)
}

/// Replaces the settings with `settings` and saves them.
///
/// A file that exists but can't be read blocks saving; that is not a failure,
/// it is reported as `saving_blocked`.
#[tauri::command]
pub async fn update_settings(
    app: AppHandle,
    settings: Settings,
) -> Result<SettingsView, CommandError> {
    let handle = app.clone();
    let (view, always_on_top_changed) = tauri::async_runtime::spawn_blocking(move || {
        let state = handle.state::<AppState>();
        let mut store = lock_store(&state.store);
        let always_on_top_changed =
            store.settings().appearance.always_on_top != settings.appearance.always_on_top;

        match store.update(settings) {
            Ok(()) | Err(SaveError::Blocked) => {
                Ok((SettingsView::from_store(&store), always_on_top_changed))
            }
            Err(error) => Err(CommandError::new(error.to_string())),
        }
    })
    .await
    .map_err(|_| CommandError::new("Couldn't save your settings. Please try again."))??;

    if always_on_top_changed {
        apply_always_on_top(&app, view.settings.appearance.always_on_top);
    }
    Ok(view)
}

/// Restores the default settings, backing up an unreadable file first.
#[tauri::command]
pub async fn reset_settings(app: AppHandle) -> Result<SettingsView, CommandError> {
    let handle = app.clone();
    let view = tauri::async_runtime::spawn_blocking(move || {
        let state = handle.state::<AppState>();
        let mut store = lock_store(&state.store);
        store
            .reset_to_defaults()
            .map_err(|error| CommandError::new(error.to_string()))?;
        Ok::<_, CommandError>(SettingsView::from_store(&store))
    })
    .await
    .map_err(|_| CommandError::new("Couldn't reset your settings. Please try again."))??;

    apply_always_on_top(&app, view.settings.appearance.always_on_top);
    Ok(view)
}

/// Lets the user choose a model file for `slot`.
///
/// The file is only returned; the settings are not touched.
#[tauri::command]
pub async fn pick_model_file(
    app: AppHandle,
    slot: SlotId,
) -> Result<Option<PickedFile>, CommandError> {
    tauri::async_runtime::spawn_blocking(move || pick_model_file_blocking(&app, slot))
        .await
        .map_err(|_| CommandError::new("Couldn't open the file picker. Please try again."))?
}

/// Asks the estimator for a new forecast.
///
/// `settings` may carry unsaved values; without them the stored settings are
/// used. Returns the generation the forecast will carry.
#[tauri::command]
pub async fn request_estimate(app: AppHandle, settings: Option<Settings>) -> u64 {
    let state = app.state::<AppState>();
    let settings = match settings {
        Some(settings) => settings.normalized(),
        None => state.settings(),
    };
    state.estimator.request(settings)
}

/// Loads the text model configured in Settings.
///
/// Errors are already written for the person using Regent; the frontend only
/// has to show them.
#[tauri::command]
pub async fn start_chat(app: AppHandle) -> Result<(), CommandError> {
    let config = {
        let state = app.state::<AppState>();
        engine_config(&state.settings())
    };
    let handle = app.clone();
    tauri::async_runtime::spawn_blocking(move || handle.state::<AppState>().chat.start(config))
        .await
        .map_err(|_| CommandError::new(CHAT_TASK_FAILED))?
        .map_err(CommandError::new)
}

/// Drops the text model and stops any reply that is streaming.
#[tauri::command]
pub async fn stop_chat(app: AppHandle) -> Result<(), CommandError> {
    let handle = app.clone();
    tauri::async_runtime::spawn_blocking(move || handle.state::<AppState>().chat.stop())
        .await
        .map_err(|_| CommandError::new(CHAT_TASK_FAILED))
}

/// Stops the reply that is currently streaming, if any.
#[tauri::command]
pub async fn cancel_generation(app: AppHandle) {
    app.state::<AppState>().chat.cancel();
}

/// The chat service's current state.
#[tauri::command]
pub async fn get_chat_state(app: AppHandle) -> ChatStatus {
    app.state::<AppState>().chat.status()
}

/// The newest stored chat messages, oldest first.
#[tauri::command]
pub async fn get_history(app: AppHandle) -> Result<Vec<ChatMessage>, CommandError> {
    let handle = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        handle.state::<AppState>().chat.history(HISTORY_READ_CAP)
    })
    .await
    .map_err(|_| CommandError::new(CHAT_TASK_FAILED))?
    .map_err(CommandError::new)
}

/// Runs one chat turn, streaming `ChatEvent`s through `on_event`.
#[tauri::command]
pub async fn send_message(
    app: AppHandle,
    text: String,
    on_event: Channel<ChatEvent>,
) -> Result<(), CommandError> {
    let system_addition = app.state::<AppState>().settings().system_prompt_addition;
    let handle = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        handle
            .state::<AppState>()
            .chat
            .send(&text, &system_addition, &mut |event| {
                let _ = on_event.send(event);
            })
    })
    .await
    .map_err(|_| CommandError::new(CHAT_TASK_FAILED))?
    .map_err(CommandError::new)
}

/// The text model configuration from the settings, or `None` when no model is
/// chosen.
fn engine_config(settings: &Settings) -> Option<EngineConfig> {
    let slot = &settings.models.text;
    slot.path.as_ref().map(|path| EngineConfig {
        model_path: path.clone(),
        gpu_layers: slot.gpu_layers,
        context_size: slot.context_size,
    })
}

/// Locks the store, ignoring a poisoned lock: the data is still usable.
fn lock_store(store: &Mutex<SettingsStore>) -> MutexGuard<'_, SettingsStore> {
    store
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Applies the always-on-top setting to the main window, ignoring a missing one.
fn apply_always_on_top(app: &AppHandle, always_on_top: bool) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
        let _ = window.set_always_on_top(always_on_top);
    }
}

/// Opens the native file dialog and resolves the user's choice.
fn pick_model_file_blocking(
    app: &AppHandle,
    slot: SlotId,
) -> Result<Option<PickedFile>, CommandError> {
    let dialog = app.dialog().file();
    let dialog = match slot {
        SlotId::Text | SlotId::Vision | SlotId::VisionProjector | SlotId::Embedding => {
            dialog.add_filter("GGUF model", &["gguf"])
        }
        SlotId::SpeechToText => dialog.add_filter("Whisper model", &["bin", "gguf"]),
        SlotId::TextToSpeech => dialog.add_filter("Piper voice", &["onnx"]),
    };

    let Some(file) = dialog.blocking_pick_file() else {
        return Ok(None);
    };
    let path = file
        .into_path()
        .map_err(|_| CommandError::new("Couldn't read the chosen file's location."))?;

    let warning = match slot {
        SlotId::TextToSpeech if !piper_config_path(&path).exists() => {
            Some(VOICE_CONFIG_WARNING.to_string())
        }
        _ => None,
    };

    Ok(Some(PickedFile {
        path: path.display().to_string(),
        warning,
    }))
}

/// The sidecar Piper reads next to a voice: `<voice>.onnx.json`.
fn piper_config_path(voice: &Path) -> PathBuf {
    let mut path = voice.as_os_str().to_owned();
    path.push(".json");
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use regent_core::estimate::{compute_estimate, Estimate, FileInfo, ModelInspector};
    use regent_core::hardware::testing::FakeProbe;
    use regent_core::hardware::{CpuInfo, GpuReport, RamInfo};

    use super::*;

    /// An inspector that answers "not found" for every file.
    struct NoFiles;

    impl ModelInspector for NoFiles {
        fn inspect(&self, _path: &Path) -> Result<FileInfo, String> {
            Err("File not found".to_string())
        }
    }

    fn probe() -> FakeProbe {
        FakeProbe::new(
            RamInfo {
                total: 32 << 30,
                available: 16 << 30,
            },
            CpuInfo {
                name: "Test CPU".to_string(),
                logical_threads: 8,
                physical_cores: Some(4),
            },
            GpuReport::default(),
        )
    }

    #[test]
    fn settings_view_json_has_the_documented_keys() {
        let view = SettingsView {
            settings: Settings::default(),
            load_error: None,
            saving_blocked: false,
            settings_path: "C:/regent/settings.toml".to_string(),
        };

        let json = serde_json::to_value(&view).expect("a view serialises");
        for key in ["settings", "load_error", "saving_blocked", "settings_path"] {
            assert!(json.get(key).is_some(), "missing `{key}` in {json}");
        }
        assert_eq!(json["load_error"], serde_json::Value::Null);
    }

    #[test]
    fn estimate_json_has_the_documented_keys() {
        let estimate: Estimate = compute_estimate(&Settings::default(), &probe(), &NoFiles);

        let json = serde_json::to_value(&estimate).expect("an estimate serialises");
        for key in ["rows", "budget", "status", "status_label"] {
            assert!(json.get(key).is_some(), "missing `{key}` in {json}");
        }
    }

    #[test]
    fn piper_config_sits_next_to_the_voice() {
        let voice = Path::new("C:/voices/en_US.onnx");
        assert_eq!(
            piper_config_path(voice),
            PathBuf::from("C:/voices/en_US.onnx.json")
        );
    }
}
