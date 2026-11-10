# 0005 — Speech-to-text helper: pipe protocol and whisper.cpp

## Status

Accepted, 2026-11-16 (spike)

## Context

Speech-to-text is a separate executable, `crates/regent-stt`, rather than part of
`regent-app`. `llama-cpp-2` and `whisper-rs` each statically vendor their own copy
of ggml; linking both into one binary collides on symbols (decision 0001, spec §3).
Isolating whisper.cpp in a helper also keeps its CPU spikes and its model load out
of the UI process.

The helper is split in two so the protocol is testable with no model, no native
library and no audio hardware (spec §3, "Tests without hardware"):

- a library (`src/lib.rs`): `protocol` and `service`;
- a binary (`src/main.rs`): argument parsing and the whisper backend in
  `src/whisper.rs`, behind the default `whisper` feature.

## Protocol

Launch:

```text
regent-stt --model <path> [--language <code|auto>] [--threads <n>]
```

Defaults: language `auto`, threads `max(1, available_parallelism − 1)`.

**stdout** is UTF-8 JSON, one object per line, flushed after each line:

```json
{"type":"ready","version":"<crate version>"}
{"type":"transcript","id":<u64>,"text":"…","duration_ms":<u64>}
{"type":"error","id":<u64>,"message":"…"}
{"type":"fatal","message":"…"}
```

`ready` is sent once the model has loaded. `error` is a bad request and the
helper keeps running. `fatal` means bad arguments or a model that failed to
load; the helper then exits with code 1. `duration_ms` is the length of the
audio, not the time transcription took.

**stdin** is binary frames, little-endian: `id: u64`, `sample_rate: u32`,
`sample_count: u32`, then `sample_count × f32` mono samples.

- Only 16,000 Hz is accepted. Other rates get an `error` and the payload is
  still consumed.
- More than 560,000 samples (35 s at 16 kHz) gets an `error`, then the payload
  is skipped in 64 KiB chunks using one stack buffer. It is never allocated,
  so a corrupt `sample_count` near `u32::MAX` cannot exhaust memory.
- Clean EOF between frames exits 0; EOF in the middle of a frame is logged to
  stderr and also exits 0.

**stderr** carries free-form log lines only. Nothing but protocol JSON may
reach stdout, so the backend calls `whisper_rs::install_logging_hooks()` at
load; with no `log`/`tracing` backend feature this discards whisper.cpp's and
ggml's own logging instead of letting it print to stdout.

The protocol implementation is `src/protocol.rs` (`Request`, `Response`,
`read_request`, `write_request`, `write_response`, `parse_response`) and the
loop is `service::run_loop`. `write_request`/`parse_response` exist for the
app-side supervisor and for tests.

## Features and runtime

- `whisper-rs = { version = "0.16.0", optional = true, default-features = false }`,
  behind `whisper = ["dep:whisper-rs"]` with `default = ["whisper"]`.
  `--no-default-features` builds the protocol library and a binary that answers
  every launch with `fatal: built without speech recognition` and exit code 1.
- All GPU backends stay off (`cuda`, `vulkan`, `metal`, `coreml`, `hipblas`,
  `intel-sycl`, `openblas`): none of their features are enabled. Whisper.cpp's
  CMake would otherwise build device code it cannot use on a machine without
  the matching toolkit.
- OpenMP stays off. `whisper-rs-sys` (0.15.0) defines `GGML_OPENMP=OFF` in its
  build script whenever its `openmp` feature is absent, so no `VCOMP140.dll` or
  `libomp*.dll` is pulled in (spec §2.6).
- Static C runtime, both sides:
  - Rust: `.cargo/config.toml` already sets `target-feature=+crt-static` for the
    MSVC targets.
  - whisper.cpp: `whisper-rs-sys` forwards every `CMAKE_*` environment variable
    as a CMake cache entry. `.cargo/config.toml` sets
    `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` and
    `CMAKE_POLICY_DEFAULT_CMP0091=NEW`, so CMake generates `/MT` for whisper.cpp
    and ggml as well. This is the same mechanism decision 0004 uses for
    llama.cpp; the two crates need separate build directories but not separate
    runtime settings.
- Each request creates a fresh `WhisperState` with greedy sampling
  (`SamplingStrategy::Greedy { best_of: 1 }`), translation off, no past context,
  and segment texts concatenated in order.

## Lifetime rules (spec §2.3)

Done in this task (helper side):

- the helper exits on stdin EOF, with no cleanup handshake required;
- bad arguments and model-load failures report `fatal` and exit 1;
- the protocol keeps the process alive after bad *requests* only, so the app
  can decide whether to retry an utterance without restarting the model.

Left to the app-side supervisor (a later task):

- ship `regent-stt.exe` with `bundle.externalBin` and resolve its path from the
  install location (environment override for development only);
- spawn with `CREATE_NO_WINDOW` so no console window flashes;
- put the child in a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so it
  dies with the app even after a crash or Task Manager kill;
- wait for `ready` with a timeout before sending audio;
- restart **once** on an unexpected exit, then surface a user-facing error;
- forward stderr to the app's log file.

## Consequences

- The helper's stdout must never be used for anything but protocol lines. New
  backend code gets the same logging treatment as whisper.cpp or it corrupts
  the stream.
- The 35 s cap is deliberate: it bounds one frame's memory and matches an
  utterance, not a recording. Longer audio must be segmented before it reaches
  the helper.
- A default `cargo build -p regent-stt` now needs CMake and LLVM/libclang on
  Windows; `--no-default-features` skips whisper.cpp entirely.
- If GPU support is added later, it must follow spec §2.6: load the backend
  dynamically and fall back to CPU, never make launch depend on a driver.
