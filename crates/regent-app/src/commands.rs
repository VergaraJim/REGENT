//! Tauri IPC commands.
//!
//! Every command must be `async`. Slow work must leave the main thread via
//! `tauri::async_runtime::spawn_blocking` so the UI never stalls.

use serde::Serialize;

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
