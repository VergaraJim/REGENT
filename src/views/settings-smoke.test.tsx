import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SettingsView } from "./SettingsView.tsx";
import type { Settings } from "../lib/api.ts";

const settings: Settings = {
  models: {
    text: { path: "C:\\models\\text.gguf", gpu_layers: 12, context_size: 8192 },
    vision: {
      model_path: null,
      projector_path: null,
      gpu_layers: 0,
      max_image_edge: 1024,
      keep_alive_secs: 300,
    },
    embedding: {
      path: null,
      gpu_layers: 0,
      context_size: 512,
      document_prefix: null,
      query_prefix: null,
    },
    stt: { path: null, language: "auto" },
    tts: { voice_path: null },
  },
  compaction: { trigger_percent: 80, target_percent: 50 },
  persist_prefix_cache: true,
  system_prompt_addition: "",
  memory: {},
  voice: { speak_replies: false, pause_mic_while_speaking: true },
  appearance: { theme: "system", always_on_top: false },
};

describe("SettingsView markup", () => {
  it("renders the sections", () => {
    const html = renderToStaticMarkup(
      <SettingsView
        onBack={() => {}}
        controller={{
          view: {
            settings,
            load_error: null,
            saving_blocked: false,
            settings_path: "C:\\settings.toml",
          },
          saveError: null,
          update: () => {},
          reset: () => {},
        }}
      />,
    );
    expect(html).toContain("Text model");
    expect(html).toContain("Appearance");
  });
});
