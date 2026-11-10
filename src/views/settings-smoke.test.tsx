import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SettingsView } from "./SettingsView.tsx";
import { MemoryContextCard } from "../components/settings/MemoryContextCard.tsx";
import type { Estimate, Settings } from "../lib/api.ts";

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

  it("renders the Memory & context card with an estimate", () => {
    const html = renderToStaticMarkup(
      <MemoryContextCard
        settings={settings}
        estimate={estimate}
        pending={false}
        onChange={() => {}}
      />,
    );
    expect(html).toContain("6.2 of 11.8 GB RAM");
    expect(html).toContain("Context size");
    expect(html).toContain("Reduced to fit this PC");
    expect(html).toContain("Use detected");
  });
});

const estimate: Estimate = {
  rows: [
    {
      slot: "text",
      file_name: "text.gguf",
      weights: 4 * 2 ** 30,
      context: 8192,
      extra: 0.8 * 2 ** 30,
      ram: 4.8 * 2 ** 30,
      vram: 0,
      kv_unknown: false,
      layer_count: 32,
      error: null,
    },
  ],
  fixed_extras: 1.4 * 2 ** 30,
  ram_used: 6.2 * 2 ** 30,
  vram_used: 2.1 * 2 ** 30,
  budget: {
    ram: 11.8 * 2 ** 30,
    ram_source: "detected",
    vram: 8 * 2 ** 30,
    vram_source: "detected",
    unified: false,
  },
  status: "ok",
  status_label: "Fits",
  context: {
    limits: {
      model_max: 32768,
      system_max: 16384,
      effective_max: 16384,
      limiting: "this_pc",
      fits: true,
    },
    requested: 32768,
    applied: { value: 16384, reason: "above_max" },
  },
  compaction: { trigger: 80, target: 50, hard_limit: 95 },
  hardware: {
    cpu: { name: "Test CPU", logical_threads: 12, physical_cores: 6 },
    ram: { total: 11.8 * 2 ** 30, available: 6 * 2 ** 30 },
    gpus: [
      {
        name: "Test GPU",
        vendor: "NVIDIA",
        kind: "dedicated",
        dedicated_vram: 8 * 2 ** 30,
      },
    ],
  },
};
