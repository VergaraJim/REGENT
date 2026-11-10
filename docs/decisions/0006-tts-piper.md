# 0006 — Text-to-speech: bundled Piper

## Status

Accepted, 2026-11-17

## Context

Regent speaks replies offline (spec §2, §5.9, §6, §7; decision 0001). Text-to-speech
must ship inside the app: never taken from `PATH`, never a system install, nothing for
the user to install. Voices are model files the user picks, so they are not part of the
app either.

## Release

We pin Piper `2023.11.14-2`, the release published at
<https://github.com/rhasspy/piper/releases/tag/2023.11.14-2>. It is the last release with
standalone binaries for all three operating systems (Windows x64, Linux x64/arm64,
macOS x64/arm64). After it, upstream moved to a GPL-only repository, whose newer
builds are not something we want to redistribute and whose layout does not match the
`piper` executable plus `espeak-ng-data` folder we ship.

Each asset is pinned by URL, size and SHA-256 in `third_party/piper/manifest.json`, one
entry per target triple. `scripts/fetch-piper.mjs` (run as `npm run fetch:piper` and
wired into `beforeDevCommand` / `beforeBuildCommand`) downloads the asset for the
target triple into a `.part` file in the OS temp folder, checks size and SHA-256
(refusing on mismatch), extracts it with the system `tar`, and copies the archive's
`piper/` folder into `crates/regent-app/resources/piper/`. It stamps `.fetched` with the
asset's SHA-256; when the stamp and the executable are present it does nothing and
touches no network. Fetched binaries are never committed: `.gitignore` ignores the
folder except its `README.md`.

## Layout

`crates/regent-app/resources/piper/` is bundled whole, as a Tauri resource mapped to
`piper/`:

```json
"resources": { "resources/piper/": "piper/" }
```

The folder holds the `piper` executable, its shared libraries (`piper_phonemize`,
`onnxruntime`, `espeak-ng`), `espeak-ng-data/` and `libtashkeel_model.ort`. Shipping the
whole folder — subfolders included — is what lets Piper's dynamic libraries load from
its own directory: the executable resolves them relative to itself, and the child
process is spawned with the Piper folder as its working directory. Development and
release therefore use the same relative path, `<resource_dir>/piper/`, and only the
folder's parent differs (`target/debug` in dev, the install directory in a bundle).

`regent-app`'s `tts` module resolves that folder (`resolve_piper`), checks the voice's
`.onnx.json` sibling (`voice_config_path`), and runs Piper with `--model`,
`--espeak_data` and `--output_raw` over pipes (`piper_command`, `synthesize_once`).
Debug builds only honour `REGENT_PIPER_DIR` as an override for development. On Windows
the child is spawned with `CREATE_NO_WINDOW` so no console window flashes.

## Licences

| Component | Licence |
|---|---|
| Piper | MIT |
| piper-phonemize | MIT |
| ONNX Runtime | MIT |
| espeak-ng | GPL-3.0 |

espeak-ng ships as a separate shared library and data folder that the Piper executable
loads; it is not linked into `regent-app`, which is what keeps the app's own licence
position unchanged (decision 0001). Our notices (`THIRD_PARTY_NOTICES` and the
`licenses/` folder) must carry the licence texts for all four and a written offer for
the espeak-ng sources, pointing at the exact release tag.

## Voices

No voice is bundled or committed. The user picks an `.onnx` file plus its `.onnx.json`;
Regent refuses a voice whose config file is missing with "This voice needs its
.onnx.json file in the same folder."

## Risks

- **macOS assets.** `piper_macos_*.tar.gz` is the least exercised of the five: it is
  unsigned, and we have not run it on either architecture. If it fails there, the
  fallback is building Piper from source in CI for that target.
- **Signing executables inside Resources.** A notarised macOS `.app` and a signed
  Windows installer want every nested executable signed. An unsigned `piper` binary
  inside `Resources/` can fail Gatekeeper or break the bundle's signature. The fallback
  is the same: build Piper from source in CI and sign it as part of the build.
- **Windows VC runtime imports.** `piper.exe` and its DLLs import `VCRUNTIME140.dll`,
  `VCRUNTIME140_1.dll` and `MSVCP140.dll` (checked with `dumpbin /dependents`), so they
  need the Visual C++ redistributable on the machine. Shipping those DLLs next to
  `piper.exe` (legacy app-local deployment) is a follow-up task; until then this is a
  gap in the "install nothing" promise for TTS on a clean Windows machine.

## Consequences

- Bumping the Piper version means editing the manifest (URL, size, SHA-256), deleting
  the fetched folder and refetching; the build refuses to proceed on a hash mismatch.
- CI must run `npm run fetch:piper` (it already does, through `beforeBuildCommand`)
  before any test that runs the engine. The `piper_smoke` test stays opt-in: it skips
  itself when the voice or the fetched engine is missing.
- Any new binary we place inside `resources/piper/` inherits the same signing and
  licence obligations as the engine itself.
