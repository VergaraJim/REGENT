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
`llama` feature of `crates/regent-app`), so it additionally needs CMake and LLVM/libclang.
`--no-default-features` builds skip llama.cpp entirely.

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
