# Regent

Regent is a small, always-on-top, private desktop assistant. Everything happens on the user's
machine: a local text model writes the replies, an optional vision model answers questions
about the screen, embedding-based long-term memory recalls past conversations, and speech
to-text and text-to-speech handle voice in and voice out. It runs locally with no network,
no cloud APIs and no accounts, on Windows first, with macOS and Linux designed in from the
start. Regent is being rebuilt as a Tauri v2 app with a Rust backend and a React + Tailwind
+ HeroUI frontend. Status: early development.

## Development (Windows)

Prerequisites:

- Rust stable with the MSVC toolchain
- Node.js 22+
- CMake
- LLVM, for libclang (set `LIBCLANG_PATH` if bindgen can't find it)
- Windows long paths enabled

The command that works today, with no models, audio hardware or GPU required:

```sh
cargo test --workspace --no-default-features
```

The default build also compiles llama.cpp in-process (text engine, behind the default
`llama` feature of `crates/regent-app`) and whisper.cpp for the `regent-stt` helper
(behind its default `whisper` feature), so it additionally needs CMake and LLVM/libclang.
`--no-default-features` builds skip both native engines entirely.

## Speech-to-text helper

`crates/regent-stt` is a separate executable so whisper.cpp's vendored ggml never has to
link into the same binary as llama.cpp. It speaks binary frames in, JSON lines out, over
stdin/stdout only; see `docs/decisions/0005-stt-helper.md` for the protocol and lifetime
rules.

```sh
cargo test -p regent-stt --no-default-features   # protocol tests, no model needed
cargo build -p regent-stt                        # compiles whisper.cpp
```

To exercise the real model once you have a GGML whisper model on disk:

```powershell
$env:REGENT_TEST_STT_MODEL="C:\path\to\ggml-base.en.bin"
cargo test -p regent-stt --test whisper_smoke -- --nocapture
```


## Text-to-speech

Replies are spoken by Piper, which ships inside the app: the pinned release is fetched
at build time into `crates/regent-app/resources/piper/` and bundled as a resource, so it
is never taken from `PATH` and nothing is installed separately. The URL, size and
SHA-256 per target triple live in `third_party/piper/manifest.json`.

```sh
npm run fetch:piper
```

`npm run tauri dev` and `npm run tauri build` run it automatically and skip the network
when the engine is already there. Fetched binaries are not committed. Voices are model
files the user picks (`.onnx` plus its `.onnx.json`); see
`docs/decisions/0006-tts-piper.md`.

To exercise the real engine, point it at a Piper voice:

```powershell
$env:REGENT_TEST_PIPER_VOICE="C:\path\to\en_US-lessac-low.onnx"
cargo test -p regent-app --no-default-features --test piper_smoke -- --nocapture
```

## Frontend development

The web frontend lives at the repository root (Vite + React + TypeScript).

```sh
npm ci
npm run dev
npm test
```

`npm run dev` serves the frontend at http://localhost:5173 for `npm run tauri dev` and
browser development; `npm test` runs the Vitest suite. Other scripts: `npm run
typecheck`, `npm run lint`, `npm run build` (outputs to `dist/`, which the Tauri crate will
bundle).

## Tauri app

The desktop app crate is `crates/regent-app` (Tauri v2). From the repository root:

```sh
npm run tauri dev
```

starts the Vite dev server on port 5173 and opens the Regent window (380×560, always
on top) against it.

```sh
npm run tauri build
```

builds the release frontend once, then the Rust backend and the installers. Run `npm ci`
and `npm run build` first so `dist/` exists and dependencies are installed.

## Models

Regent ships no model weights. Every model has its own licence, and users are responsible
for complying with it.
