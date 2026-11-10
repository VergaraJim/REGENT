# 0004 — Text engine: llama.cpp via llama-cpp-2

## Status

Accepted, 2026-11-10

## Context

Text, vision and embeddings run in-process on llama.cpp (spec §3). The `llama-cpp-2`
crate (0.1.161, with `llama-cpp-sys-2` 0.1.161) builds llama.cpp from source with CMake
and binds it with bindgen, both of which are installed on this machine. The spike sets up
a default `llama` feature of `regent-app` with a minimal load-and-generate function and an
opt-in test.

## Decision

- Crate: `llama-cpp-2 = { version = "0.1.161", optional = true, default-features = false }`,
  behind `llama = ["dep:llama-cpp-2"]` and `default = ["llama"]` in
  `crates/regent-app/Cargo.toml`.
- Features turned **off** (by dropping the crate's defaults):
  - `openmp` — llama.cpp would be built with OpenMP, which adds `VCOMP140.dll`/
    `libomp*.dll` at runtime. Spec §3 demands no extra runtime DLLs. llama.cpp's own
    thread pool provides intra-op parallelism, so nothing is lost for our use.
  - every GPU backend (`cuda`, `vulkan`, `metal`, `opencl`, `rocm`, `mkl`) — the app is
    CPU-only right now; turning them off also avoids NVIDIA/Vulkan build dependencies
    on the dev machine.
  - `common` — only needed for chat-template helpers we do not use yet.
- Static C runtimes, both sides:
  - Rust side: `.cargo/config.toml` already sets `target-feature=+crt-static` for the
    msvc targets (decision 0003).
  - llama.cpp (C/C++) side: CMake does not see those rustflags. `llama-cpp-sys-2`'s
    build script reads `LLAMA_STATIC_CRT=1` and then calls
    `cmake ... config.static_crt(true)` (`MultiThreaded`, i.e. /MT). On its own, however,
    that only lands as `-MT` extra compiler flags, and the msbuild generator keeps the
    default dynamic runtime in the generated `.vcxproj` — the first build still produced a
    binary depending on `MSVCP140.dll` (found with `dumpbin /dependents`). The sys build
    script also forwards every `CMAKE_*` environment variable as a CMake cache entry, so
    `.cargo/config.toml` sets, next to `LLAMA_STATIC_CRT = "1"`:
    `CMAKE_MSVC_RUNTIME_LIBRARY = "MultiThreaded"` and
    `CMAKE_POLICY_DEFAULT_CMP0091 = "NEW"`. After that, `dumpbin /dependents
    target\debug\regent-app.exe` shows only Windows system DLLs: no `VCRUNTIME140*.dll`,
    no `MSVCP140*.dll`, no `VCOMP140*.dll`, no `libomp*.dll` (spec §2.6).
- The engine code is in `src/engine/`, llama part behind `#[cfg(feature = "llama")]`,
  with a minimal `smoke_generate(model_path, prompt, max_tokens)` (CPU, 0 GPU layers,
  default mmap, 512-token context, threads = max(1, available_parallelism − 1), BOS
  tokenisation, greedy sampling until end-of-generation or the budget). The llama.cpp
  backend is initialised at most once per process via a `OnceLock` (`llama_backend_init`
  must never run twice).
- Opt-in test: `crates/regent-app/tests/llama_smoke.rs`, run with the
  `llama` feature:
  `REGENT_TEST_TEXT_MODEL=C:\path\to\model.gguf cargo test -p regent-app --test llama_smoke`.
- Inference is CPU only (`with_n_gpu_layers(0)`, default memory mapping), and CPU-only
  in the CMake build too because no GPU backend feature is enabled.

## GPU plan

The spec (§2.6) plans to load ggml GPU backends **dynamically at runtime**, so a machine
without a GPU or with a broken driver never blocks launch. When we take that step, we
enable `llama-cpp-2`'s `dynamic-backends` feature (which builds the GPU backend DLLs out
of the main link and loads them with `load_backends_from_path`): missing files then just
mean CPU inference. Until then, CPU is the only path and no GPU DLL ships in the bundle.

## Consequences

- Because llama.cpp and whisper.cpp each vendor their own ggml, they cannot be linked into
  the same binary. Speech-to-text runs in the separate `regent-stt` helper process via
  `whisper-rs` (decision 0001), keeping the two ggmls apart.
- A default `cargo check --workspace` now needs CMake and LLVM/libclang on Windows;
  `--no-default-features` skips it (README documents both).
