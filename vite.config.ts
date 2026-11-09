import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { configDefaults } from "vitest/config";

// https://vite.dev/config/
export default defineConfig({
  clearScreen: false,
  plugins: [react(), tailwindcss()],
  // Vite options tailored for Tauri development.
  // 1. prevent vite from obscuring rust errors
  envPrefix: ["VITE_", "TAURI_ENV_"],
  server: {
    // Tauri expects a fixed port, fail if that port is not available
    port: 5173,
    strictPort: true,
    watch: {
      // ignore the Rust workspace and lane state
      ignored: ["**/crates/**", "**/target/**", "**/.soma/**"],
    },
  },
  build: {
    // Tauri uses Chromium on Windows and WebKit on macOS and Linux
    target: process.env.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari13",
  },
  test: {
    exclude: [...configDefaults.exclude, ".soma/**", "crates/**", "target/**"],
  },
});
