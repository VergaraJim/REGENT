# 0001 — Stack overview

## Status

Accepted, 2026-11-10

## Context

Regent is a private desktop assistant where every model — text, vision, embeddings,
speech-to-text and text-to-speech — runs locally with no network. It is a Tauri v2 app:
Rust backend, React + TypeScript + Vite frontend with Tailwind CSS and HeroUI. Windows is
the primary target, with macOS and Linux designed in from the start. Spec §2 requires the
app to be self-contained: no redistributable runtimes, no servers, no listening sockets,
and every helper process shipped, started, supervised and stopped by the app. The existing
egui prototype is a behavioural reference only.

## Decision

| Area | Choice |
|---|---|
| **App** | Tauri v2. Frontend: React + TypeScript + Vite, Tailwind CSS, HeroUI. Inter font and lucide icons are bundled; nothing is fetched at runtime. |
| **Crates** | `regent-core`: pure logic. `regent-app`: Tauri backend. `regent-stt`: speech-to-text helper process. |
| **Text, vision, embeddings** | llama.cpp via the `llama-cpp-2` crate, GGUF models, in-process, behind a default `llama` feature of `regent-app`. |
| **Model metadata** | Our own GGUF header reader in `regent-core`. |
| **Speech-to-text** | whisper.cpp via `whisper-rs`, inside the `regent-stt` helper, talking over stdin/stdout pipes. Reason: llama.cpp and whisper.cpp each vendor ggml and would clash at link time. |
| **Text-to-speech** | Piper's official prebuilt release. It is fetched at build time with a pinned URL and SHA-256, shipped as a resource folder, and run as a child process over pipes. espeak-ng (GPL-3.0) stays a separate executable. Voices are model files the user picks; they are never bundled. |
| **Database** | SQLite via `rusqlite` (`bundled`). |
| **Vector search** | Exact cosine scan in Rust over L2-normalised vectors. |
| **Audio** | `cpal` + `rubato`. |
| **Settings** | TOML via `serde` + `toml`. |
| **Paths** | Via the `dirs` crate: settings `<config_dir>/regent/settings.toml`; database `<data_local_dir>/regent/regent.db`; models `<data_local_dir>/regent/models`; prefix cache `<cache_dir>/regent/kv`. |
| **C runtime** | Linked statically on Windows (`+crt-static`). |
| **Packaging** | Per-user NSIS installer with the WebView2 offline installer; AppImage and .deb; macOS .app/.dmg. Identifier: `com.vergara-regent.desktop`. |
| **CI** | GitHub Actions on Windows, Ubuntu and macOS (arm64 and x64). The repository is public. |
| **History** | No code or data is imported from the earlier egui prototype. |

## Consequences

- Every helper process must follow spec §2.3: shipped inside the bundle, started and
  supervised by the app, communicating over stdin/stdout pipes, dying with the app.
- Every new native dependency must pass the dependency check in spec §2 (no hidden
  redistributables, no listening sockets) before it is kept.
- Because llama.cpp and whisper.cpp cannot be linked into one binary, any change that
  adds another ggml-vendoring engine must keep the in-process vs helper-process split
  explicit.
- Piper is a build-time artifact with a pinned URL and SHA-256; the build must refuse to
  proceed on mismatch.
