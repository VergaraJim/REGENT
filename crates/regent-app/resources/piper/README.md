# Piper resources

This folder receives the pinned Piper text-to-speech engine at build time. Its
contents are fetched, never committed: `.gitignore` ignores everything here except
this file.

Run:

```sh
npm run fetch:piper
```

That downloads the asset for the current target triple from
`third_party/piper/manifest.json`, verifies its size and SHA-256, extracts it, and
copies the archive's `piper/` folder here (the `piper` executable, its shared
libraries, and `espeak-ng-data/`). It writes a `.fetched` stamp holding the asset's
SHA-256; when the stamp and the executable are still present the script does nothing
and touches no network.

`npm run tauri dev` and `npm run tauri build` run it automatically through
`beforeDevCommand` / `beforeBuildCommand`, and the whole folder ships as a Tauri
resource at `piper/` next to the app.

On Windows, the fetch step also copies `msvcp140.dll`, `vcruntime140.dll` and
`vcruntime140_1.dll` here from the local Visual Studio install (the `VC\Redist\MSVC`
folder), so Piper runs on a machine without the Visual C++ redistributable. Set
`REGENT_VC_REDIST_DIR` to a folder that holds those three DLLs to use a different
source.

Voices (`.onnx` plus `.onnx.json`) are not part of this folder: the user picks them.
