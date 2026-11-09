# Prompt — Rebuild Regent as a self-contained Tauri desktop app

> Paste everything below the line into the coding agent that will build the app. The existing
> Rust project (`regent`, egui UI) is the reference for behaviour. If the agent has that repo
> checked out, point it at the path in **§3**. Otherwise the spec here is enough to build from.

---

## 1. What you are building

**Regent** is a small, always-on-top, private desktop assistant. Everything happens on the
user's machine:
- a local text model;
- optional vision;
- embedding-based long-term memory;
- voice in (speech-to-text);
- voice out (text-to-speech).

A first version exists in Rust with an egui UI. Rebuild it as a **Tauri v2** app (Rust backend,
web frontend). **Windows is the primary target.** Design for macOS and Linux from the start.

**You choose the stack inside Tauri.** That includes the inference engines, model formats, speech
engines, database, vector index, audio I/O and frontend framework. This document specifies
behaviour, limits and constraints, not libraries. Pick whatever best satisfies §2 on all three
OSes. Record each choice and its reasoning in a short decision record under `docs/decisions/`.

## 2. Non-negotiable requirement: everything is inside the app

The app must run on a fresh machine with **nothing else installed and no network**, once its
model files are on disk. Each rule below is an acceptance criterion.

1. **Local models only.** All inference runs on the user's CPU/GPU from model files on the
   user's disk. That covers text, vision, embeddings, speech-to-text and text-to-speech. No
   cloud APIs, no accounts, no API keys.
2. **No external services or prerequisites.** The user never installs, starts or configures
   anything else. That rules out:
   - a local inference server;
   - a Python or Node runtime;
   - Docker;
   - a database server;
   - a separately installed speech engine or voice data;
   - a C/C++ runtime redistributable;
   - a GPU toolkit;
   - a separately downloaded WebView runtime.

   If your stack needs any of these, either do the work in-process or ship it with the app
   under rule 3.
3. **If a feature needs a helper process, the app ships it, starts it, supervises it and stops
   it.** For every helper:
   - **Shipping.** It goes inside the installer or bundle: Tauri `bundle.externalBin` for
     executables, `bundle.resources` for folders. Its path is resolved from the app's own
     install location. An environment-variable override is for development only.
   - **Communication.** It talks over **stdin/stdout pipes**, not network ports. If a port is
     truly unavoidable, bind `127.0.0.1` only, use an ephemeral port and a random per-launch
     auth token. Never bind `0.0.0.0`.
   - **Startup.** It starts lazily and must print a "ready" handshake within a timeout.
   - **Crashes.** If it dies, restart it **once**, then show a clear, user-facing error. Never
     retry in a loop.
   - **Lifetime.** It dies with the app, **including when the app crashes or is killed**. It
     exits on stdin EOF. On Windows, also assign it to a Job Object with
     `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.
   - **Windows console.** Spawn it with `CREATE_NO_WINDOW` so no console window flashes.
   - **Logging.** Send its stderr to the app's log file.
4. **No network at runtime.** That means no telemetry, analytics, crash upload, update checks,
   remote fonts, CDN scripts or remote images.
   - Set a strict Tauri CSP with no remote origins.
   - Bundle every font and asset.
   - The **only** network access allowed is a **user-initiated** model download (§5.1). It must
     be optional: picking a local file always works.
5. **No listening sockets.** Frontend↔backend traffic uses Tauri IPC only.
6. **Self-contained binaries.** No redistributable runtime may be required.
   - Link the C/C++ runtime statically, or ship it inside the app folder.
   - Do this consistently across Rust and any vendored native code. A runtime mismatch between
     the two is a common link failure.
   - A GPU acceleration path must **never** stop the app from launching on a machine without
     that GPU or driver. Load GPU code dynamically or fall back to CPU.

**Acceptance tests for §2.** Automate them in CI where possible.
- **Clean machine.** On a clean Windows 10/11 VM (no runtime redistributables, no dev tools, no
  Python): install, add models, **disable networking**, then use every feature.
- **Dependencies.** `dumpbin /dependents` (Windows), `otool -L` (macOS) and `ldd` (Linux) list
  only OS libraries for every shipped binary.
- **No sockets.** While the app runs, `netstat -ano` shows nothing in `LISTENING` for its PIDs.
  Outbound connections appear only during a user-started download.
- **No leftovers.** Kill the main process from Task Manager. No helper process survives.
- **Cut-down PATH.** Run with `PATH` set to `C:\Windows\System32;C:\Windows`. Everything still
  works.

## 3. Reference implementation (behaviour, not stack)

The existing repo is at `C:\Users\Jim Vergara\Projects\Soma\regent`.
- It is the source of truth for behaviour where this document is silent. Where they conflict,
  this document wins.
- Reuse its code only where it fits the stack you chose.
- Read `README.md`, `docs/decisions/0001–0005` and these files under `crates/regent-core/src/`:
  `footprint.rs`, `context.rs`, `memory.rs`, `prompt.rs`, `memory_index.rs`, `tools/`,
  `audio/vad.rs` and `engines/stt_post.rs`. The algorithms there are pure and well tested.

Lessons from the reference that apply to any stack:
- **Duplicate symbols.** Two native engines that each statically vendor the same underlying
  library may not link into one binary. Check this early with a spike. If they clash, isolate
  one in a helper process (§2.3).
- **Hidden runtime dependencies.** A native library's default features can add one, such as an
  OpenMP DLL. Re-run the §2 dependency check after every dependency change.
- **Tests without hardware.** Tests must run with no models, no audio hardware and no GPU. Put
  engines, microphone, speaker and clock behind traits or factories, and inject fakes.

## 4. Architecture

```
┌──────────────── Tauri app process ──────────────────┐
│ WebView (frontend)  ⇄  Tauri IPC  ⇄  Rust backend    │
│                                                      │
│  Service worker: conversation, prompt + cache,       │
│    compaction, tools, memory indexing, database      │
│  Audio capture → voice activity detection            │
│  Speech output: synthesis → playback                 │
│  Estimator: model metadata + memory arithmetic       │
└──────────┬───────────────────────────────────────────┘
           │ pipes only (for any engine that must run out of process)
   bundled helper process(es), started and supervised by the app
```

- **One service worker owns the slow, stateful work.** The UI sends commands and receives
  events.
  - **Commands:** start, stop, send text, cancel generation, set listening, get/set settings,
    request estimate, pick file, download model.
  - **Events:** state changed, token, message saved, generation stats, compaction
    started/finished/failed, tool ran, memory/indexing status, listening changed, mic level,
    audio error, STT error, vision loading/error.
  - **Transport:** stream tokens over `tauri::ipc::Channel`. Send state changes with
    `AppHandle::emit`. Throttle mic level to at most about 15 reports a second, and only around
    speech.
- **Never block the main thread.** Inference, disk access, model-metadata reads and hardware
  detection all run on worker threads.
- **Frontend.** Any framework. Nothing may be fetched at runtime. The theme follows the system
  (light/dark), with a manual override.

## 5. Functional spec

### 5.1 Model slots ("model choosing")

Each role is a separate slot in settings. The user picks a file with the native file dialog, or
downloads one from a curated catalog.

| Slot | Purpose | Per-slot settings |
|---|---|---|
| **Text model** | Writes the replies | GPU offload amount (none → all), context size (§5.3) |
| **Vision model** | Answers questions about a screenshot. May need a companion file, such as a projector; keep all required files together as one slot. | GPU offload, max image edge (default 1024 px), keep-alive (default 120 s, then unload) |
| **Embedding model** | Vectors for long-term memory | GPU offload, context (default 2048), document/query prefix overrides |
| **Speech-to-text** | Transcribes utterances | Language (`auto` by default; English-only models want `en`) |
| **Text-to-speech voice** | Reads replies aloud | Voice file(s), kept together with any companion config |

Rules:
- **A feature is offered only when its model is configured.**
  - No embedding model: no `search_memory` tool.
  - No complete vision slot: no `look_at_screen` tool.
  - No STT model: no voice input. Typed chat still works, and the UI says why and links to
    settings.
- **Engine defaults.** Memory-map weights where the engine supports it. Use about one thread per
  physical core, leaving one core for the rest of the app while STT runs alongside.
- **Model catalog and downloader.** The reference has only placeholders, so this needs finishing.
  - The catalog is a fixed list per slot. Each entry has a name, file(s), URL, **SHA-256**,
    exact size, and licence name and URL.
  - **Do not invent URLs or hashes.** Leave clearly marked `TODO` entries for the owner to fill
    in. An incomplete entry is "not downloadable".
  - Downloads run off the UI thread. Write `<name>.part`, verify size and SHA-256, then rename.
    A failed download leaves nothing behind.
  - Files go to `<user data dir>/regent/models`.
  - Show size and licence before downloading (§7).
- **(New)** Grey out or warn on catalog entries whose estimated footprint (§5.2) is `TooBig` for
  this machine. Model choice should be guided by hardware.

### 5.2 Hardware detection and RAM/VRAM forecast

Everything here is computed **before** any model loads. It is pure arithmetic over plain data,
unit tested with fake hardware and fake model metadata.

**Detection:**
- Total and available RAM, CPU name and logical thread count.
- GPUs with dedicated VRAM, for any vendor, on Windows and Linux.
- Apple Silicon: **unified memory**. GPU-offloaded bytes count against RAM. Respect the OS's
  recommended GPU working-set limit.
- Integrated or unknown GPUs contribute no VRAM. Never guess a figure.
- Re-read the RAM figures for each estimate. Detect the GPU list once.

**Budget:**
- RAM: the user's override, else available RAM. If the OS reports 0 available, use installed
  RAM.
- VRAM: the user's override, else the largest dedicated VRAM found.
- The user can override both, in GB, with a "Use detected" reset.

**Model metadata.** Read it from the model file's header without loading the weights. That costs
a few KB, even for a 20 GB file. Skip large arrays such as vocabularies, and bound every length
so a corrupt file can't trigger huge allocations. Fields:
- architecture;
- trained context length;
- layer count;
- embedding width;
- attention head count;
- KV head count (fallback: head count);
- key and value head dims (fallback: width ÷ heads);
- sliding window.

**Per-model footprint:**
- **Weights** = file size on disk.
- **KV cache** = layers × n_ctx × kv_heads × (key_dim + value_dim) × bytes_per_element.
  - Round **up**.
  - Bytes per element depend on the KV cache type: 2 for 16-bit, proportionally less for
    quantised caches (about 1.0625 for 8-bit blocks and 0.5625 for 4-bit blocks). Use exact
    integer arithmetic.
  - Missing metadata means "unknown" (0), never "free".
- **Overhead** = the larger of 256 MiB and 5% of the weights.
- **Models without usable metadata** (often STT and TTS): 1.2 × file size, no KV.
- **GPU split:** gpu_fraction = offloaded layers ÷ layers. That share of (weights + KV) goes to
  VRAM. Overhead always stays in RAM.
- **Vision model and companion files:** weights + overhead, no KV of their own.
- **Total:** all RAM parts + fixed extras (audio buffers, the app itself). VRAM parts are summed
  separately.

**Fit status.** Compute it per pool; the worse of RAM and VRAM wins.

| Share of budget used | Status |
|---|---|
| Under 70% | **Ok** ("Fits comfortably") |
| 70–90% | **Tight** |
| Over 90% | **TooBig** |

Needing VRAM when the machine has none is TooBig.

**(New)** After loading, show the measured usage next to the forecast: process RSS, plus GPU
memory where it can be queried.

### 5.3 Context window sizing

- **Usable memory** = 85% of the RAM budget, minus the other models' footprints, minus the text
  model's weights and overhead. The 15% held back covers the OS, other apps and allocations the
  estimate can't see.
- **`system_max`** = the largest multiple of **256** tokens whose KV cache fits in usable
  memory. Find it by binary search, since KV grows linearly. Cap it at the engine's maximum
  context value.
- **`model_max`** = the trained context length, if known.
- **`effective_max`** = min(`model_max`, `system_max`).
- **Limiting factor**, shown to the user: Model / This PC / Equal / Unknown.
- **Below 256:** if `effective_max` is under 256, the setup doesn't fit. Say so, and never load a
  zero context.
- **Clamping the user's requested context:**
  1. Round it down to a multiple of 256.
  2. Raise it to at least 256.
  3. Cap it at `effective_max`. The cap is used as-is, even when it isn't on the 256 step.

  Report why it changed: within limits, raised to minimum, rounded, above max, or no room.
- **The "Memory & context" card** shows:
  - per-model rows (Weights, Context and Extra, in GB);
  - the total against the budget, with the fit label;
  - a context slider with "Model max X · This PC max Y" and a "Reduced to fit this PC" hint;
  - the compaction trigger and target sliders (§5.5);
  - the memory override inputs.
- **Running the estimate.** It runs on a worker thread.
  - Only the newest pending request is computed.
  - Generation numbers drop stale results, so dragging a slider never builds a backlog.
- **Assumption.** The estimate assumes a 16-bit KV cache, the honest upper bound. **(New,
  optional)** Use the sliding window to tighten the KV estimate for sliding-window models.

### 5.4 Conversation engine and cache reuse

- **The prompt must be a strict prefix of the next turn's prompt.** Order it like this:
  1. a constant system prompt plus the user's optional addition from settings, with no dates or
     other runtime values;
  2. the tool section: a constant calling protocol plus one line per tool, sorted by name;
  3. the rolling summary, if there is one;
  4. the turns, oldest first.

  Anything dynamic (time, retrieved text, status) goes only in the newest message.
- **Chat template.** Render with the model's own template, with a sensible fallback.
- **Reuse computation across turns.** This is the difference between a responsive CPU assistant
  and seconds of stall per turn. Pick an engine that supports it.
  1. Tokenise the new prompt.
  2. Compare it with the tokens already cached for that conversation. Compare **tokens, never
     text**.
  3. Discard the cache after the shared prefix and process only the rest.
- **Background work never evicts the main cache.** Summaries and similar jobs must not throw
  away the main conversation's cache. Use a separate sequence, slot or context.
- **Persisted prefix cache** (if the engine can save and restore state):
  - Save the cache state of the fixed prefix (system prompt + tools) under
    `<cache dir>/regent/…`.
  - Key it on: model path, size and mtime, context size, KV type, chat template, exact prefix
    text and engine version.
  - Load a state only on an exact key match with the expected file length.
  - Write to a temp file, then rename. Keep at most 3 states.
  - Save in the first idle window after a reply, never before one. On by default.
- **Reply reserve:** 512 tokens. They are kept out of the prompt budget, and the same number caps
  one generation.
- **History read cap:** at most 512 stored messages are read back per turn. Compaction bounds the
  actual prompt.
- **Generation.** It streams token by token and can be cancelled at any point. It reports stats:
  prompt tokens, reused, processed, generated, and tokens per second.

### 5.5 Automatic context compaction

- **Thresholds**, derived from n_ctx:

  | Threshold | Formula |
  |---|---|
  | trigger | n_ctx × 80% − 512 |
  | target | n_ctx × 45% − 512 |
  | hard limit | n_ctx − 512 |

  Trigger and target are user-adjustable. They are validated as `target < trigger < 100`; invalid
  values fall back to 80/45.
- **Working memory** = the latest summary + the messages after it + the fixed prefix's token
  cost. Sizes come from the model's own tokeniser, plus a measured per-message template cost
  (default 4).
- **When the prompt passes the trigger:**
  - Keep the **newest 6 messages verbatim**.
  - Fold everything older, together with the previous summary, into a new summary.
  - Cap the summary at **40% of the target**, with a minimum of 32 tokens.
- **The summarizer:**
  - Runs on the background cache slot at temperature 0.2.
  - Has a constant system prompt: "merge previous summary + new messages; keep names, facts,
    preferences, decisions, open tasks; drop small talk; plain prose; summary only".
  - Its output is rejected if empty, over the cap, or cancelled. Cancellation is checked every
    token.
- **When it runs:** in the **idle window** after a reply, and only if no user turn is waiting.
  If a prompt would exceed the hard limit, it runs **synchronously, before generating**.
- **History is never edited.**
  - Messages are append-only, enforced by the database where possible.
  - A pass only appends a summary; the newest summary wins.
  - If summarizing fails, drop the oldest turns **from the prompt only**.
- **UI:** a subtle "Compacting…" state.

### 5.6 Long-term memory (vector mapping)

- **Storage.** One local embedded database file in the user data dir. No server.
- **Data model:**
  - **conversations.**
  - **messages:** role (system/user/assistant/tool), content, token count, timestamp, source
    (typed or voice). **Append-only.**
  - **summaries:** the last message id covered, content, token count, timestamp. Never edited;
    the newest wins.
  - **embeddings**, one row per chunk: message id, chunk index, model id, dimension, vector.
    These rows are the source of truth.
  - **index metadata:** which model and dimension the vector index was built for.
  - Schema changes go through versioned migrations.
- **Vector index:**
  - It lives inside the app, either in the same file or derived from it. No server.
  - It is built for the current embedding model's dimension. Changing the model or dimension
    rebuilds it.
  - Old vectors are reported as stale and re-indexed in the background, never mixed in.
  - If the accelerated index is unavailable on a platform, fall back to an exact brute-force
    cosine scan with the same ranking.
- **Chunking:**
  - About 256 tokens per chunk with 20% overlap (estimated at 4 chars per token).
  - A message matches a query through its best chunk, so search returns messages, not
    fragments.
- **Embedding conventions:**
  - Store vectors L2-normalised, so cosine is a dot product.
  - Some retrieval models expect different marker text before stored text and before queries.
    Auto-detect this from the model's metadata or name where known, with settings overrides.
  - Store the model id with every vector.
- **Indexing never slows a reply.**
  - It runs in idle windows, in batches of 4 messages, checking for a new user turn after each
    batch.
  - At most 64 messages are queued in memory; the database is the real queue.
  - After 3 consecutive failures, indexing pauses.
  - On every start, rebuild the queue from the database: user and assistant messages without a
    vector from the current model. That resumes after a crash and re-indexes after a model
    change.
  - `tool` and `system` rows are **never** embedded.
- **UI:** the footer shows indexed/total counts and an "indexing…" hint.

### 5.7 Tool calls (no native function-calling needed)

Tools must work with small local models that have no function-calling support. Use a plain-text
protocol parsed from the stream. The reference format is
`<tool_call>{"name":"…","arguments":{…}}</tool_call>`.

- **Streaming parser:**
  - Hides the block from the visible reply, holding back trailing characters that could be the
    start of a tag.
  - Stops generation at the closing tag.
  - Drops an unterminated block at the end of the reply, but never drops ordinary text.
- **Validation.** Check the JSON and the arguments against the tool's parameter schema. An
  unknown tool, a missing or unknown argument, or an out-of-range value gets a `tool` message
  explaining the error.
- **Budget: at most 3 calls per user turn.**
  - A rejected call counts toward the 3.
  - The last allowed result tells the model to answer now.
  - The closing round runs with tools off.
- **Storage.** Store the text written before a call as `assistant`, the result as `tool`, and the
  continuation as `assistant`. The prompt is rebuilt from the database, so it replays exactly what
  happened.
- **UI.** One muted chip under the assistant message, e.g. "Searched memory" or "Looked at
  screen". There is no tool panel.
- **`search_memory(query, limit? 1..8, default 4)`:**
  1. Embed the query with the query marker.
  2. Search all conversations (64 candidates).
  3. Exclude messages already in the working-memory window, and all `tool`/`system` rows.
  4. Load each hit together with the message before it.
  5. Output one dated line per message, e.g. `[2026-11-08] user: …`. Collapse whitespace and cut
     each snippet to at most 300 chars. Cap the whole result at about 600 tokens.
  6. If nothing has been indexed yet, say so in one line.
- **`look_at_screen(question)`:** see §5.10.

### 5.8 Voice in (always listening)

- **Capture:**
  - Use the default input device, on a dedicated thread.
  - Open it with a handshake and a timeout, so "no device" and "permission denied" are reported
    straight away.
  - Convert to the format STT needs (e.g. mono, 16 kHz) with a streaming resampler. There must be
    no clicks at buffer boundaries.
  - The real-time callback never blocks and never makes unbounded allocations.
- **Mic switch.** The mic is on while the assistant runs. Switching it off **closes** the device,
  not just mutes it.
- **Mic failures** (unplugged, denied) are reported once, and listening turns off. No retry loop.
- **Voice activity detection** sits behind an interface, so a model-based detector can replace
  it. Reference defaults (energy based):
  - 20 ms frames, RMS compared against an adaptive threshold.
  - Noise floor = the **minimum** frame energy over the last 2 s. Frames inside speech never
    feed the noise floor.
  - 200 ms start debounce, 300 ms pre-roll, 700 ms end hangover, 30 s utterance cap.
- **STT:**
  - Runs off the conversation thread, so a reply keeps streaming while the next sentence becomes
    text.
  - If it runs out of process, apply §2.3: start it at Start if a model is set, 120 s load
    timeout, restart once, kill it on Stop.
  - Audio goes over as one payload per utterance, never per frame.
  - One utterance is in flight at a time and up to 3 more wait in a queue. The rest are dropped.
- **Post-filter:**
  - Trim whitespace.
  - STT models often emit stock phrases on silence: "Thank you.", "Thanks for watching", "Please
    subscribe", blank-audio markers. Drop these **only** when the utterance was short (under
    2.5 s) or quiet (RMS under 0.01).
  - Drop results under 2 chars.
- **Result.** The transcript is stored and answered exactly like typed text. It is marked as
  voice, with a small mic icon on the bubble.

### 5.9 Voice out

- **Bundling.** The TTS engine and its voice data ship with the app (§2). In release builds they
  resolve from the app bundle, never from `PATH` or a system install.
- **Clean-up.** A streaming sentence splitter removes markdown, code blocks and tool markup before
  anything is spoken.
- **Playback.** Speak sentence by sentence, so the first words play while the rest is being
  synthesized. Playback runs on its own thread.
- **Stop** cuts the audio immediately, dropping queued samples.
- **Mic pause.** Pause the mic while speaking and resume 400 ms after playback ends. Setting:
  `pause_mic_while_speaking`, on by default.
- **Spoken replies** are a setting too: `speak_replies`, on by default.

### 5.10 Vision (still to build; stubbed in the reference)

- **What the reference has:** the vision engine interface, the `look_at_screen` tool, the vision
  loading/error events, the keep-alive logic and screen capture. The engine itself only returns
  "not available".
- **What to implement:**
  - Run vision in a **separate context** from the text model, so screenshots never enter the
    main cache.
  - Capture the primary monitor and resize it so the longest edge is at most the max image edge.
    **Pixels are never written to disk.**
  - Cap answers at about 300 tokens.
  - Unload the model after the keep-alive time with no requests.
- **Permissions.** Handle a denied Wayland screen-capture portal and macOS Screen Recording
  permission with a clear message.

### 5.11 Settings and storage

- **The settings file** is human-readable, in the user config dir.
  - Every field has a default, so old or partial files load.
  - An unreadable or invalid file is **never overwritten**. The app runs on defaults and shows
    the error.
- **Fields:**
  - the five model slots (§5.1);
  - context size (default 4096);
  - compaction trigger % (80) and target % (45);
  - persist prefix cache (on);
  - system-prompt addition;
  - RAM and VRAM overrides;
  - `pause_mic_while_speaking` and `speak_replies`;
  - theme: system, light or dark.
- **Existing users.** The reference stores data here, under `regent` in the platform's user
  dirs:
  - settings: `%APPDATA%\regent\settings.toml` (Linux: `~/.config/regent/settings.toml`);
  - database: `<data dir>/regent/regent.db`;
  - models: `<data dir>/regent/models`;
  - prefix cache: `<cache dir>/regent/kv`.

  On first launch, find that data and import or keep it: settings, conversation history and
  model paths. Nothing a user had should be lost.

### 5.12 UI

- **Window:**
  - Small and always-on-top: 380×560, minimum 320×420, with the standard OS title bar.
  - Always-on-top can be toggled. On Wayland the compositor may ignore it; document that.
- **Header:**
  - Start/Stop.
  - A microphone switch with a live level meter.
  - A gear button that opens settings.
- **Chat:**
  - Streaming assistant bubbles; voice turns get a mic icon.
  - Tool chips.
  - The "Compacting…" state.
  - A footer with memory-indexing status.
- **Settings:**
  - Model slots with file pickers and download buttons.
  - Speech language.
  - The system-prompt addition.
  - Audio toggles and theme.
  - The **Memory & context** card (§5.2–5.3).
  - The **Licences** screen (§7).
- **Error messages** are written for end users and say what to do. For example: "The speech
  helper is missing from this install — reinstall Regent." No engine jargon.

## 6. Platform and packaging

- **Windows (primary):**
  - NSIS per-user installer, so no admin rights are needed. MSI is optional.
  - Set `bundle.windows.webviewInstallMode` to `offlineInstaller` (or `fixedRuntime` for a
    portable build), so WebView2 never needs a separate download.
  - No redistributables (§2.6).
  - Unsigned builds trigger SmartScreen. Plan for code signing.
- **macOS:**
  - GPU acceleration on Apple Silicon, with unified-memory budgeting (§5.2).
  - Add the microphone usage description and the hardened-runtime audio-input entitlement.
  - Vision needs Screen Recording permission.
  - Sign every bundled binary and library, and notarize the app.
- **Linux:**
  - Build an AppImage and a `.deb`.
  - Document the build dependencies of the stack you chose.
  - Wayland caveats: always-on-top may be ignored, and screen capture goes through the portal.
- **GPU acceleration:**
  - Use whatever each OS supports best, with a CPU fallback.
  - The GPU path must never block launch (§2.6).
  - At minimum, ship CPU plus one GPU path per OS.
- **Build pipeline:**
  - Build helpers per target triple and register them with Tauri `externalBin`.
  - Any third-party binary or data file fetched at build time is pinned by URL + SHA-256. The
    build script refuses to guess either value.
  - CI matrix: Windows, macOS (arm64 and x64), Linux.
  - Run the §2 acceptance tests on the bundles CI produces.

## 7. Licensing

- **The app** is Apache-2.0. Ship `LICENSE` and `NOTICE`.
- **Audit everything you ship.** That includes Rust crates, npm packages, native libraries,
  bundled executables and data files (voice data, phonemizer dictionaries, fonts, icons).
  - In CI, generate a third-party notices file that covers all of them.
  - Fail CI on any licence that is not on an allow-list.
- **Copyleft (GPL/LGPL/AGPL)** is common in speech tooling.
  - Prefer permissive alternatives.
  - If you do ship a copyleft component, record in a decision record why it is compatible (for
    example, it is a separate executable, not linked code).
  - Include its licence text and meet its source-offer obligations.
- **Fonts and icons:** ship their licences (e.g. OFL).
- **Models: ship no model weights.**
  - Every catalog entry shows its licence name and link before download.
  - Many model licences are non-commercial or require attribution. The user is responsible for
    complying; say so in the UI and the README.
- **In the app:** a **Licences** screen lists every third-party notice and works offline.

## 8. What the reference does NOT do yet

- **Vision inference:** stubbed (§5.10).
- **Default model catalog:** placeholders only, so "Download all" is disabled.
- **GPU:**
  - The app is CPU-only today.
  - VRAM detection is NVIDIA-only, behind a build flag.
  - The offload settings and the estimator's VRAM split already exist.
- **macOS:** not supported at all yet.
- **Voice activity detection:** a model-based detector is planned behind the existing interface.
- **Helper processes on Windows:** they are spawned without hiding the console window. Fix this
  as described in §2.3.

## 9. Order of work

1. **Scaffold and spike.**
   - Set up Tauri v2.
   - Choose the stack, and prove that every engine builds, links and runs on all three OSes.
     Record this in decision records.
   - Set up the static or app-local runtime, the CSP and a minimal capabilities file.
   - Add CI that builds on all three OSes.
2. **Settings and the Memory & context card.** Hardware detection, metadata reading, footprint,
   context limits and the coalesced estimator. Pure logic; no model needed.
3. **Service bridge.** Start, stop, send and cancel; token streaming; replaying history on start;
   the chat UI.
4. **Compaction and cache reuse,** including the persisted prefix cache.
5. **Embeddings and memory.** The vector index, idle indexing and `search_memory`.
6. **Voice in and voice out,** including the mic pause.
7. **Vision** and `look_at_screen`.
8. **Model catalog and verified downloads,** with fit-aware recommendations.
9. **Ship.** Packaging, licence notices, and the §2 acceptance tests on each OS.

## 10. Definition of done

- Every item in §2 passes on Windows. The same holds for macOS and Linux bundles if they are
  built.
- `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo test --workspace` pass without models, audio hardware or a GPU. Tests that need a model
  are opt-in, through environment variables pointing at model files.
- The third-party notices are complete and the licence check passes (§7).
- With models configured and networking off, a user can:
  - type to the assistant;
  - talk to it and hear the reply;
  - ask about something from a past conversation and get it recalled;
  - ask about what is on screen;
  - chat past the context limit without errors, as compaction kicks in;
  - see an accurate RAM/VRAM forecast before anything loads.
