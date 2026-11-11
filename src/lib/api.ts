/**
 * Typed wrappers over Regent's Tauri IPC commands and events.
 *
 * The Rust structs in `crates/regent-core/src/settings.rs` and
 * `crates/regent-core/src/estimate.rs` serialise with their field names
 * unchanged (snake_case) and with enums renamed to snake_case strings, so the
 * types below mirror that JSON exactly. Optional fields are left out of the
 * JSON when they are `null` on the Rust side.
 */
import { Channel, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { UnlistenFn } from "@tauri-apps/api/event";

/* ------------------------------------------------------------------ */
/* Settings                                                            */
/* ------------------------------------------------------------------ */

export type Theme = "system" | "light" | "dark";

export interface TextSlot {
  path?: string | null;
  gpu_layers: number;
  context_size: number;
}

export interface VisionSlot {
  model_path?: string | null;
  projector_path?: string | null;
  gpu_layers: number;
  max_image_edge: number;
  keep_alive_secs: number;
}

export interface EmbeddingSlot {
  path?: string | null;
  gpu_layers: number;
  context_size: number;
  document_prefix?: string | null;
  query_prefix?: string | null;
}

export interface SttSlot {
  path?: string | null;
  language: string;
}

export interface TtsSlot {
  voice_path?: string | null;
}

export interface ModelSlots {
  text: TextSlot;
  vision: VisionSlot;
  embedding: EmbeddingSlot;
  stt: SttSlot;
  tts: TtsSlot;
}

export interface CompactionSettings {
  trigger_percent: number;
  target_percent: number;
}

export interface MemoryOverrides {
  ram_gb?: number | null;
  vram_gb?: number | null;
}

export interface VoiceSettings {
  speak_replies: boolean;
  pause_mic_while_speaking: boolean;
}

export interface Appearance {
  theme: Theme;
  always_on_top: boolean;
}

export interface Settings {
  models: ModelSlots;
  compaction: CompactionSettings;
  persist_prefix_cache: boolean;
  system_prompt_addition: string;
  memory: MemoryOverrides;
  voice: VoiceSettings;
  appearance: Appearance;
}

/** The settings plus the state of the file they were read from. */
export interface SettingsView {
  settings: Settings;
  load_error: string | null;
  saving_blocked: boolean;
  settings_path: string;
}

/* ------------------------------------------------------------------ */
/* Model files                                                         */
/* ------------------------------------------------------------------ */

export type SlotId =
  | "text"
  | "vision"
  | "vision_projector"
  | "embedding"
  | "speech_to_text"
  | "text_to_speech";

/** A model file the user chose; `warning` is set when it needs attention. */
export interface PickedFile {
  path: string;
  warning: string | null;
}

/* ------------------------------------------------------------------ */
/* Memory estimate                                                     */
/* ------------------------------------------------------------------ */

export type FitStatus = "ok" | "tight" | "too_big";

export type BudgetSource = "override" | "detected" | "installed_fallback" | "none";

export type GpuKind = "dedicated" | "integrated" | "unknown";

export type LimitingFactor = "model" | "this_pc" | "equal" | "unknown";

export type ClampReason =
  | "within_limits"
  | "rounded"
  | "raised_to_minimum"
  | "above_max"
  | "no_room";

export interface ModelRow {
  slot: SlotId;
  file_name: string;
  weights: number;
  context: number;
  extra: number;
  ram: number;
  vram: number;
  kv_unknown: boolean;
  layer_count: number | null;
  error: string | null;
}

export interface MemoryBudget {
  ram: number;
  ram_source: BudgetSource;
  vram: number;
  vram_source: BudgetSource;
  unified: boolean;
}

export interface ContextLimits {
  model_max: number | null;
  system_max: number | null;
  effective_max: number | null;
  limiting: LimitingFactor;
  fits: boolean;
}

export interface ClampedContext {
  value: number | null;
  reason: ClampReason;
}

export interface ContextReport {
  limits: ContextLimits;
  requested: number;
  applied: ClampedContext;
}

export interface CompactionThresholds {
  trigger: number;
  target: number;
  hard_limit: number;
}

export interface CpuInfo {
  name: string;
  logical_threads: number;
  physical_cores: number | null;
}

export interface RamInfo {
  total: number;
  available: number;
}

export interface GpuInfo {
  name: string;
  vendor: string | null;
  kind: GpuKind;
  dedicated_vram: number | null;
}

export interface HardwareSummary {
  cpu: CpuInfo;
  ram: RamInfo;
  gpus: GpuInfo[];
}

/** The whole memory forecast. */
export interface Estimate {
  rows: ModelRow[];
  fixed_extras: number;
  ram_used: number;
  vram_used: number;
  budget: MemoryBudget;
  status: FitStatus;
  status_label: string;
  context: ContextReport | null;
  compaction: CompactionThresholds | null;
  hardware: HardwareSummary;
}

/** The payload of the `estimate-ready` event. */
export interface EstimateReady {
  generation: number;
  estimate: Estimate;
}

/** The event the backend emits whenever a forecast finishes. */
export const ESTIMATE_READY_EVENT = "estimate-ready";

/* ------------------------------------------------------------------ */
/* Chat                                                                */
/* ------------------------------------------------------------------ */

export type ChatRole = "user" | "assistant";

/** Where the chat engine is in its lifecycle. */
export type ChatState =
  | "stopped"
  | "loading"
  | "ready"
  | "generating"
  | "compacting"
  | "error";

/** One stored turn in the conversation. */
export interface ChatMessage {
  id: number;
  role: ChatRole;
  text: string;
  from_voice: boolean;
  created_at: number;
}

/** The payload of the `chat-state` event and of `get_chat_state`. */
export interface ChatStateEvent {
  state: ChatState;
  error: string | null;
}

export interface GenerationStats {
  prompt_tokens: number;
  /** Prompt tokens whose KV cache was reused instead of recomputed. */
  reused_tokens: number;
  /** Prompt tokens the model actually processed for this turn. */
  processed_tokens: number;
  generated_tokens: number;
  tokens_per_second: number;
}

/** One event streamed back over the `send_message` channel. */
export type ChatEvent =
  | { type: "compacting" }
  | { type: "token"; text: string }
  | { type: "done"; message_id: number; stats: GenerationStats }
  | { type: "cancelled"; message_id: number | null }
  | { type: "error"; message: string };

/** The event the backend emits whenever the chat engine changes state. */
export const CHAT_STATE_EVENT = "chat-state";

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

/** Reads the settings and the state of their file. */
export function getSettings(): Promise<SettingsView> {
  return invoke<SettingsView>("get_settings");
}

/**
 * Saves `settings`.
 *
 * A settings file that exists but can't be read blocks saving; the returned
 * view then has `saving_blocked` set instead of the call failing.
 */
export function updateSettings(settings: Settings): Promise<SettingsView> {
  return invoke<SettingsView>("update_settings", { settings });
}

/** Restores the defaults, backing up an unreadable file first. */
export function resetSettings(): Promise<SettingsView> {
  return invoke<SettingsView>("reset_settings");
}

/** Opens the native picker for `slot`; resolves to `null` when cancelled. */
export function pickModelFile(slot: SlotId): Promise<PickedFile | null> {
  return invoke<PickedFile | null>("pick_model_file", { slot });
}

/**
 * Asks for a new forecast and returns its generation.
 *
 * Pass unsaved settings to preview a change; omit them to use the stored ones.
 */
export function requestEstimate(settings?: Settings): Promise<number> {
  return invoke<number>("request_estimate", { settings: settings ?? null });
}

/** Calls `cb` for every finished forecast. Remember to call the unlistener. */
export function onEstimate(
  cb: (ready: EstimateReady) => void,
): Promise<UnlistenFn> {
  return listen<EstimateReady>(ESTIMATE_READY_EVENT, (event) => {
    cb(event.payload);
  });
}

/* ------------------------------------------------------------------ */
/* Chat commands                                                       */
/* ------------------------------------------------------------------ */

/** Loads the text model from the settings and gets the engine ready. */
export function startChat(): Promise<void> {
  return invoke<void>("start_chat");
}

/** Unloads the engine and clears the in-memory model. */
export function stopChat(): Promise<void> {
  return invoke<void>("stop_chat");
}

/** Stops the current reply, saving whatever text was already produced. */
export function cancelGeneration(): Promise<void> {
  return invoke<void>("cancel_generation");
}

/** Reads the engine's current state. */
export function getChatState(): Promise<ChatStateEvent> {
  return invoke<ChatStateEvent>("get_chat_state");
}

/** Reads the stored conversation, oldest first. */
export function getHistory(): Promise<ChatMessage[]> {
  return invoke<ChatMessage[]>("get_history");
}

/**
 * Sends `text` and calls `onEvent` for every streamed token and for the final
 * `done`, `cancelled` or `error` event. Resolves when the turn ends.
 */
export function sendMessage(
  text: string,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const channel = new Channel<ChatEvent>();
  channel.onmessage = onEvent;
  return invoke<void>("send_message", { text, onEvent: channel });
}

/** Calls `cb` for every engine state change. Remember to call the unlistener. */
export function onChatState(
  cb: (event: ChatStateEvent) => void,
): Promise<UnlistenFn> {
  return listen<ChatStateEvent>(CHAT_STATE_EVENT, (event) => {
    cb(event.payload);
  });
}

/* ------------------------------------------------------------------ */
/* Stale-result guard                                                  */
/* ------------------------------------------------------------------ */

export interface LatestOnly {
  /** Records the newest generation this UI is waiting for. */
  track(generation: number): void;
  /** Whether a result for `generation` is still the newest one. */
  accept(generation: number): boolean;
}

/**
 * Keeps the newest generation the UI asked for, so results older than that can
 * be ignored when they arrive out of order.
 */
export function createLatestOnly(): LatestOnly {
  let newest = 0;
  return {
    track(generation: number): void {
      if (generation > newest) {
        newest = generation;
      }
    },
    accept(generation: number): boolean {
      return generation >= newest;
    },
  };
}
