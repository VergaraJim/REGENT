# 0002 — Frontend stack and tooling

## Status

Accepted, 2026-11-09

## Context

Regent needs a Windows-first desktop UI. The window is small (about 380×560 pixels), so the
interface must stay quiet and readable. The app is offline-only by design (spec §2): no CDN,
no remote fonts, no telemetry — every asset must be bundled. The Rust/Tauri side arrives in a
later task; this decision records the frontend toolchain installed at the repo root.

## Decision

| Area | Choice | Why |
|---|---|---|
| **React** | 19.3.1 | Current stable; HeroUI v3 targets React 19 natively. |
| **Vite** | 8.3.6 (template-generated ^8.3.x) | Standard bundler for the Tauri v2 workflow; supports the `TAURI_ENV_*` env prefix and fixed-port dev server. |
| **TypeScript** | ~5.9.3 | The Vite template ships the new native TypeScript 7 preview, but typescript-eslint still caps peer support below 6.x, so we pinned the latest stable 5.x for a working toolchain. |
| **Tailwind CSS** | 4.3.5 with `@tailwindcss/vite` | HeroUI v3 is built on Tailwind CSS v4; no `tailwind.config.js` needed, configuration lives in CSS. |
| **HeroUI** | 3.2.7 (`@heroui/react` + `@heroui/styles`, framer-motion 14.11.1 peer) | Tailwind v4–based component library; no `<Provider>` wrapper is required in v3. Styles are imported as `@import "@heroui/styles"` after `@import "tailwindcss"` in `src/index.css`. Dark mode is class-based: `.dark` on `<html>`, which the theme hook toggles. |
| **Icons** | lucide-react 1.70.0 | Small tree-shaken SVG icons, bundled offline. |
| **Font** | @fontsource-variable/inter 5.3.0 | Inter bundled via npm, imported in `src/main.tsx`; no remote font fetch. |

Offline-only rule: `index.html` references no remote URL and contains no inline `<script>`;
`src/` must never fetch remote resources. All colours come from HeroUI theme tokens
(`bg-background`, `text-foreground`, `text-default-500`, `border-divider`); no custom palette.

Licences to include in the third-party notices later: Inter — SIL Open Font License 1.1
(OFL-1.1); lucide — ISC.

The `npm ci` development workflow: `npm ci`, then `npm run dev` (port 5173, strict), `npm run
build` (`tsc -b && vite build`, output in `dist/`), `npm test` (Vitest) and `npm run lint`.

## Consequences

- HeroUI's animations set inline styles, so the Tauri CSP will need
  `style-src 'unsafe-inline'`.
- The Tauri crate must point `frontendDist` at `../dist` and `devUrl` at
  `http://localhost:5173` (the dev server is strict-port and fails if 5173 is taken).
- HeroUI v3 needs no provider wrapper, but if one is added later it must not re-enable
  network features.
- The theme currently defaults to `"system"` in `src/App.tsx`; the settings view will supply
  a persisted preference later.
- typescript-eslint peer support excludes TypeScript ≥ 6.x and 7.x; revisit the pin if that
  changes.
