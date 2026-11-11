# Manual testing: compaction, memory and voice

This guide checks the parts that automated tests cannot: real models, a real
microphone, real speakers and a real restart. Every check is a **Do** line
followed by an **Expect** line. Do them in order; later stages assume the
earlier ones passed.

Code, not this page, is the source of truth. If a label or message differs, the
app is right and this page is stale.

## Before you start

**Do:** close Regent, then back up the database:

```sh
# Linux
cp ~/.local/share/regent/regent.db ~/.local/share/regent/regent.db.bak
```

```powershell
# Windows PowerShell
Copy-Item "$env:LOCALAPPDATA\regent\regent.db" "$env:LOCALAPPDATA\regent\regent.db.bak"
```

**Expect:** `regent.db.bak` exists next to `regent.db`. The database is schema
v3 now; an older build refuses it with "Regent's chat history was made by a
newer version of Regent." The backup is how you go back.

### Where Regent keeps its files

| What | Linux | Windows |
| --- | --- | --- |
| Settings | `~/.config/regent/settings.toml` | `%APPDATA%\regent\settings.toml` |
| Data folder | `~/.local/share/regent` | `%LOCALAPPDATA%\regent` |
| Database | `~/.local/share/regent/regent.db` | `%LOCALAPPDATA%\regent\regent.db` |
| Logs | `~/.local/share/regent/logs` | `%LOCALAPPDATA%\regent\logs` |
| Speech helper log | `~/.local/share/regent/logs/regent-stt.log` | `%LOCALAPPDATA%\regent\logs\regent-stt.log` |
| Prefix cache | `~/.cache/regent/kv` | `%LOCALAPPDATA%\regent\kv` |
| Models (default) | `~/.local/share/regent/models` | `%LOCALAPPDATA%\regent\models` |

On macOS the settings live under `~/Library/Application Support/regent`, the
data under `~/Library/Application Support/regent` too, and the prefix cache
under `~/Library/Caches/regent/kv`.

### Run the app in development

**Do:**

```sh
npm ci
npm run tauri dev
```

**Expect:** the Regent window opens (380×560, always on top). `beforeDevCommand`
runs `npm run fetch:piper` first, so the Piper engine is downloaded into
`crates/regent-app/resources/piper/` on the first run and reused after that.

### Build the speech helper once

**Do:**

```sh
cargo build -p regent-stt
```

**Expect:** `target/debug/regent-stt` (Windows: `regent-stt.exe`) is built. It
needs CMake and LLVM/libclang, because whisper.cpp is compiled from source. The
helper must sit next to the app binary in `target/debug`, which is exactly
where this command puts it. In debug builds only, `REGENT_STT_PATH=/path/to/regent-stt`
overrides the location; release builds ignore it.

### Model files you need

| Part | File |
| --- | --- |
| Text chat | a GGUF text model (for example a small instruct model) |
| Memory | a GGUF embedding model, such as `nomic-embed-text` or `bge-small` |
| Voice in | a Whisper ggml model, for example `ggml-base.en.bin` |
| Voice out | a Piper voice, `<name>.onnx` **plus** its `<name>.onnx.json` in the same folder |

Regent ships no model weights; each model has its own licence.

## Opt-in automated checks

These run real engines. They are skipped, by printing `skipped: …`, when their
model variable is missing, so they never fail on a machine without models.
Run them from the repository root.

```sh
# Text engine smoke test (default `llama` feature). Not #[ignore]d.
REGENT_TEST_TEXT_MODEL=/path/to/text.gguf \
  cargo test -p regent-app --test llama_smoke -- --nocapture

# Chat engine: reply, cache reuse and summarising. Three tests are #[ignore]d.
REGENT_TEST_MODEL=/path/to/text.gguf \
  cargo test -p regent-app --test llama_chat_smoke -- --ignored --nocapture

# Embedding engine: related texts must be closer than unrelated ones. #[ignore]d.
REGENT_TEST_EMBED_MODEL=/path/to/nomic-embed-text.gguf \
  cargo test -p regent-app --test llama_embed_smoke -- --ignored --nocapture

# Whisper transcription. Builds whisper.cpp, so CMake and libclang are needed.
REGENT_TEST_STT_MODEL=/path/to/ggml-base.en.bin \
  cargo test -p regent-stt --test whisper_smoke -- --nocapture

# Piper synthesis. Needs the fetched engine; --no-default-features skips llama.cpp.
REGENT_TEST_PIPER_VOICE=/path/to/en_US-lessac-low.onnx \
  cargo test -p regent-app --no-default-features --test piper_smoke -- --nocapture
```

`REGENT_TEST_PIPER_DIR` overrides the Piper folder for the last test; it
defaults to `crates/regent-app/resources/piper`.

On Windows PowerShell, set the variable first and then run the same cargo
command, for example:

```powershell
$env:REGENT_TEST_MODEL="C:\models\text.gguf"
cargo test -p regent-app --test llama_chat_smoke -- --ignored --nocapture
```

## Stage 4: Compaction and cache reuse

**Do:** start Regent with a text model, send a message and press **Stop** while
the reply is streaming.

**Expect:** the conversation stays on screen and the line "Regent is stopped.
Press Start to continue." appears under it. **Start** resumes.

**Do:** open **Settings → Memory & context** and drag **Context size** down to
512 tokens. Drag **Compact at** to 50% and **Compact down to** to 20%, then go
back, press **Stop** and **Start** again.

**Expect:** the sliders read `512 tokens`, `50%` and `20%`, and the helper line
under "Compact down to" reads "When the chat reaches 50% of the context, older
messages are summarised down to 20%." (A target at or above the trigger shows
"Must be lower than \"Compact at\"." instead.)

**Do:** chat until a reply first shows **Compacting…** (a placeholder in the
reply bubble, and briefly a "Compacting…" spinner line above the composer).

**Expect:** the turn continues after the summary instead of failing with a
context error, and later turns keep working past the old context limit.

**Do:** inspect the newest summary row:

```sh
sqlite3 ~/.local/share/regent/regent.db \
  "SELECT id, covers_through, created_at FROM summaries ORDER BY id DESC LIMIT 1;"
```

```powershell
sqlite3 "$env:LOCALAPPDATA\regent\regent.db" "SELECT id, covers_through, created_at FROM summaries ORDER BY id DESC LIMIT 1;"
```

**Expect:** one row. `covers_through` is the id of the newest message folded
into it. More rows appear as compaction runs again.

**Do:** send two similar questions in a row after pressing **Start**, and
compare how quickly the second reply starts.

**Expect:** the second turn is noticeably faster, because the system prompt and
earlier history are served from the engine's prefix cache. (The UI does not
print token counts; compare wall-clock time.)

**Do:** make sure the prefix cache is on — it is `persist_prefix_cache = true`
by default in `settings.toml`; there is no Settings switch for it yet. Send one
message, wait for the reply, then look at the cache folder:

```sh
ls ~/.cache/regent/kv
```

```powershell
dir "$env:LOCALAPPDATA\regent\kv"
```

**Expect:** at most three `*.state` / `*.toml` pairs. The cache is pruned to the
three newest entries. Restart the app, send a message again, and the first reply
is faster than a cold start.

## Stage 5: Long-term memory

**Do:** in **Settings → Models → Embedding model**, press **Choose…** and pick
a GGUF embedding model. Go back, press **Stop**, then **Start**.

**Expect:** under the composer the muted line counts up, for example "Memory: 3
of 40 messages saved · indexing…", then settles to "Memory: 40 of 40 messages
saved".

**Do:** count the stored vectors:

```sh
sqlite3 ~/.local/share/regent/regent.db \
  "SELECT COUNT(*), COUNT(DISTINCT message_id) FROM embeddings;"
```

```powershell
sqlite3 "$env:LOCALAPPDATA\regent\regent.db" "SELECT COUNT(*), COUNT(DISTINCT message_id) FROM embeddings;"
```

**Expect:** both counts are greater than zero, and the distinct message count
matches the "X" in the memory line. `SELECT model_id, COUNT(*) FROM embeddings
GROUP BY model_id;` shows the model id, which includes the model file name, its
size and a hash of the document prefix.

**Do:** compact the old part of a conversation (Stage 4), then ask about a
detail you know is in an old, now-summarised message.

**Expect:** the reply answers from memory and carries the **Searched memory**
chip under the assistant bubble. Restart the app: it is still a single bubble
with the same chip, because tool rows are folded back into their reply.

**Do:** in **Settings → Models → Embedding model → Options**, change **Document
prefix** (the placeholder is "Automatic"), then **Stop** and **Start**.

**Expect:** the memory line returns to indexing and re-embeds every message.
The embeddings table then holds a second `model_id`: the document prefix is
part of the model id, so old vectors are never mixed with new ones. (Rows are
append-only; the old model's rows stay.)

**Do:** remove the embedding model with its **Remove** (X) button, then
**Stop** and **Start**, and ask an old question.

**Expect:** no memory line under the composer, no "Searched memory" chip, and
`search_memory` is not offered to the model. Typed chat still works.

## Stage 6: Voice

### Setup

**Do:** in **Settings → Models** pick a **Speech-to-text** model and set its
**Language** (an English-only Whisper model needs English), then pick a
**Voice** (Piper `.onnx`). Go back and press **Start**.

**Expect:** with a voice whose `.onnx.json` is missing, the picker shows "This
voice needs its .onnx.json file in the same folder." With both models set, the
mic button shows a spinner ("Loading speech model…") and then a live mic; while
you speak, the small level meter beside it moves and disappears when you stop.
A text model is also required: **Start** loads the text model first.

### Voice turns

**Do:** speak one short sentence and stop.

**Expect:** a user bubble appears with the small mic icon next to the text, and
Regent answers it like a typed turn.

**Do:** stay silent, then cough once.

**Expect:** no turn appears. Whisper's stock phrases ("Thank you.", "Thanks for
watching", `[BLANK_AUDIO]` and similar) are dropped when the utterance was short
or quiet, so a cough does not create a "Thank you." message.

### Spoken replies

**Do:** leave **Read replies aloud** on and **speak** a question that needs a few
sentences.

**Expect:** the reply is spoken sentence by sentence as it streams, not only
after it finishes.

**Do:** type a question instead.

**Expect:** the reply appears as text only; nothing is spoken, even with Read
replies aloud on.

**Do:** speak a question that needs a few sentences and watch the mic while
Regent speaks.

**Expect:** it shows the muted "Microphone paused while Regent speaks" state,
then resumes about 400 ms after the audio ends; the level meter comes back.

**Do:** speak a question that needs a few sentences, then while Regent is
speaking, press **Stop speaking** (the crossed-speaker button that appears in
the header only while speech is playing).

**Expect:** the audio cuts off immediately.

**Do:** speak a question that needs a few sentences and repeat with the
composer's **Cancel** button, with **Stop**, and by sending a new message
mid-speech.

**Expect:** all three cut the speech at once.

### Settings

**Do:** turn **Read replies aloud** off and speak a question.

**Expect:** nothing is spoken; the reply is text only.

**Do:** turn it back on and turn **Pause microphone while speaking** off.

**Expect:** the mic stays live while Regent speaks (it shows the live mic and
meter, not the paused state), so it can hear you over the reply.

### Mic switch

**Do:** click the live mic to turn it off, then check the OS microphone
indicator (for example the orange dot in the menu bar, or the mic icon in the
Windows privacy tray).

**Expect:** the device is closed and the OS indicator goes off.

**Do:** click the mic again.

**Expect:** it reopens and the level meter moves when you speak.

### Failures

**Do:** with the mic live, unplug the microphone (or disable it in the OS).

**Expect:** exactly one red line appears above the memory line — for example
"The microphone stopped working. Turn the mic on again when it's back." — and
the mic turns off. There is no retry loop and no repeated error line.

**Do:** quit Regent, rename `regent-stt` (`regent-stt.exe`) next to the app
binary so it cannot be found, and start Regent with a speech model chosen.

**Expect:** the mic button shows "The speech helper is missing from this
install — reinstall Regent." and opens Settings when clicked. Rename it back.

**Do:** while Regent and the helper are running, quit Regent, then check for a
surviving process:

```sh
pgrep -a regent-stt
```

```powershell
Get-Process regent-stt -ErrorAction SilentlyContinue
```

**Expect:** no output. Dropping the app closes the helper's stdin, waits
briefly and kills it if needed.

**Do:** open the helper's log after using voice:

```sh
tail -n 50 ~/.local/share/regent/logs/regent-stt.log
```

**Expect:** the helper's stderr is appended there. The file exists after the
speech model has been loaded at least once.

### Without a speech model

**Do:** remove the speech-to-text model, then press **Start**.

**Expect:** the mic button shows a muted mic-off icon and the tooltip "Voice
input needs a speech model. Open Settings." Clicking it opens Settings. Typed
chat still works normally.

## Troubleshooting

**`npm run tauri dev` crashes in GTK/WebKit on Linux.** Try forcing X11:

```sh
GDK_BACKEND=x11 npm run tauri dev
```

**"The speech helper is missing from this install."** Build it with
`cargo build -p regent-stt` and check that `regent-stt` sits next to the app
binary in `target/debug`. In debug builds only, point at it with
`REGENT_STT_PATH`.

**No microphone or no speakers.** The app says so in plain words: "No
microphone was found…", "Regent isn't allowed to use the microphone…", "The
microphone couldn't be opened…" or "No speakers or headphones were found."
Check the OS privacy settings and whether another app holds the device, then
toggle the mic again.

**Piper is missing.** Run `npm run fetch:piper` (it also runs before
`npm run tauri dev` and `npm run tauri build`). It downloads the pinned release
into `crates/regent-app/resources/piper/` and skips the network when it is
already there. In debug builds `REGENT_PIPER_DIR` can point at another folder.

**A voice file is rejected.** Pick the `.onnx`; its `.onnx.json` must be in the
same folder and keep the same base name (`en_US-lessac-low.onnx.json`).
