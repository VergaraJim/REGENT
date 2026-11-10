# 0003 — Security and packaging

## Status

Accepted, 2026-11-10

## Context

Spec §2 requires Regent to be self-contained and private: no cloud calls, no listening
sockets, no redistributable runtimes. The app is a Tauri v2 window hosting the React
frontend on the local machine. The packaging target is Windows first (NSIS), with macOS
(.app/.dmg) and Linux (.deb/AppImage) built from the same config.

## Decision

### Content Security Policy

The production CSP (`app.security.csp`, object form) is:

- `default-src 'self'` — everything loads from the bundled assets only.
- `script-src 'self'` — no inline scripts beyond the Tauri IPC nonce; `withGlobalTauri`
  is off.
- `style-src 'self' 'unsafe-inline'` — HeroUI animations and Tailwind user styles set
  inline styles on elements. This is the one deliberate relaxation: accepting inline
  styles, with no remote styles, is the cost of using HeroUI (recorded in 0002).
- `img-src 'self' data: blob:` — `data:`/`blob:` cover generated and in-memory images;
  no remote images.
- `font-src 'self' data:` — Inter Variable and lucide icons are bundled; `data:` is a
  fallback for font embedding.
- `connect-src ipc: http://ipc.localhost` — the only allowed connection endpoints are
  Tauri's IPC scheme (and its sidecar form, which Chromium needs); no network hosts.
- `object-src 'none'`, `base-uri 'none'`, `frame-src 'none'`, `form-action 'none'` —
  no plugins, no base-URL takeover, no iframes, no form posts.

**`dangerousDisableAssetCspModification: ["style-src"]`**: by default Tauri injects a
nonce into `script-src` and `style-src`. When a nonce is present, browsers drop
`'unsafe-inline'`, so HeroUI's inline styles would be rejected. Disabling the
modification for `style-src` only keeps `'unsafe-inline'` effective. `script-src` still
gets the nonce — inline scripts remain blocked.

`app.security.devCsp` exists only for development. It differs in two ways:

- `script-src` adds `'unsafe-inline'` so the Vite dev client (which injects inline
  scripts for the error overlay) works;
- `connect-src` adds `'self'` and `ws://localhost:5173` for the Vite HMR websocket.

Both `devCsp` and the Vite dev server (localhost:5173) are development conveniences.
Release builds read the bundled `dist/` assets directly over the custom protocol, open
no listening sockets, and use Tauri IPC only.

### Capability and commands

`capabilities/default.json` grants only `core:default` to the `main` window. The dialog
plugin is registered but called from Rust, so the frontend receives no dialog
permission; every future plugin capability must be added explicitly and minimally.

Every Tauri command must be `async` (documented on `crates/regent-app/src/commands.rs`),
and any slow work must leave the main thread via `tauri::async_runtime::spawn_blocking`,
so model inference or disk I/O never blocks window event handling.

### Packaging

- Windows: NSIS installer with `installMode: "currentUser"` — installs into the user's
  profile, no admin rights required.
- WebView2 `webviewInstallMode: "offlineInstaller"` bundles the WebView2 runtime so no
  separate download happens at install time.
- The C runtime is linked statically: `.cargo/config.toml` sets
  `target-feature=+crt-static` for `x86_64-pc-windows-msvc` and
  `aarch64-pc-windows-msvc`, so the app depends on no `VCRUNTIME140.dll`.
- Application identifier: `com.vergara-regent.desktop`.
- Icons are generated once from `crates/regent-app/icons/source.svg` with
  `npx tauri icon` and committed.

## Consequences

- Any new CSP relaxation needs a new entry here with the reason.
- Any frontend-visible plugin needs its own capability entry; nothing is granted
  wholesale.
- Release builds must remain socket-free; any helper that wants a socket is a design
  error.
