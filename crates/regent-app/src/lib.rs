mod commands;
mod state;

pub mod audio;
pub mod engine;
pub mod tts;

use regent_core::paths::AppPaths;
use tauri::Manager;
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

use crate::state::AppState;

/// The only window's label, as configured in `tauri.conf.json`.
const MAIN_WINDOW: &str = "main";

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let Some(paths) = AppPaths::detect() else {
                app.dialog()
                    .message("Regent couldn't find your user folders.")
                    .title("Regent")
                    .kind(MessageDialogKind::Error)
                    .blocking_show();
                app.handle().exit(1);
                return Ok(());
            };

            let state = AppState::new(paths, app.handle().clone());
            let always_on_top = state.settings().appearance.always_on_top;
            app.manage(state);

            if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
                let _ = window.set_always_on_top(always_on_top);
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::app_info,
            commands::get_settings,
            commands::update_settings,
            commands::reset_settings,
            commands::pick_model_file,
            commands::request_estimate,
            commands::start_chat,
            commands::stop_chat,
            commands::send_message,
            commands::cancel_generation,
            commands::get_chat_state,
            commands::get_memory_status,
            commands::get_history,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Regent");
}
