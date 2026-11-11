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

export type ChatRole = "user" | "assistant" | "tool";

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
  /** The tool that produced a `tool` row, e.g. `search_memory`; null elsewhere. */
  tool_name?: string | null;
  /** The raw call that produced a `tool` row; null elsewhere. */
  tool_call?: string | null;
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
  | { type: "tool"; name: string }
  | { type: "done"; message_id: number; stats: GenerationStats }
  | { type: "cancelled"; message_id: number | null }
  | { type: "error"; message: string };

/** The event the backend emits whenever the chat engine changes state. */
export const CHAT_STATE_EVENT = "chat-state";

/** Where the memory index is in its lifecycle. */
export type MemoryState = "off" | "ready" | "indexing" | "paused";

/** The payload of the `memory-status` event and of `get_memory_status`. */
export interface MemoryStatus {
  state: MemoryState;
  /** Messages already embedded into the index. */
  indexed: number;
  /** Messages the index should hold. */
  total: number;
  error: string | null;
}

/** The event the backend emits whenever the memory index changes state. */
export const MEMORY_STATUS_EVENT = "memory-status";

/** Where the vision engine is in its lifecycle. */
export type VisionState = "off" | "idle" | "loading" | "ready";

/** The payload of the `vision-status` event and of `get_vision_status`. */
export interface VisionStatus {
  state: VisionState;
  error: string | null;
}

/** The event the backend emits whenever the vision engine changes state. */
export const VISION_STATUS_EVENT = "vision-status";

/* ------------------------------------------------------------------ */
/* Voice                                                               */
/* ------------------------------------------------------------------ */

/** Where the speech-to-text engine is in its lifecycle. */
export type VoiceState = "off" | "loading" | "ready" | "error";

/** The payload of the `voice-status` event and of `get_voice_status`. */
export interface VoiceStatus {
  state: VoiceState;
  /** Whether the microphone is capturing audio. */
  listening: boolean;
  /** Whether the mic is paused because Regent is speaking. */
  paused: boolean;
  error: string | null;
}

/** The event the backend emits whenever the voice input changes state. */
export const VOICE_STATUS_EVENT = "voice-status";

/** The payload of the `mic-level` event; `level` is 0.0 to 1.0. */
export interface MicLevel {
  level: number;
}

/** The event the backend emits with the microphone's current loudness. */
export const MIC_LEVEL_EVENT = "mic-level";

/** The payload of the `voice-transcript` event. */
export interface VoiceTranscript {
  text: string;
}

/** The event the backend emits for every finished speech transcript. */
export const VOICE_TRANSCRIPT_EVENT = "voice-transcript";

/** The payload of the `speaking` event. */
export interface SpeakingPayload {
  speaking: boolean;
}

/** The event the backend emits when spoken output starts or stops. */
export const SPEAKING_EVENT = "speaking";

/** The payload of the `voice-output-error` event. */
export interface VoiceOutputError {
  message: string;
}

/** The event the backend emits when spoken output fails. */
export const VOICE_OUTPUT_ERROR_EVENT = "voice-output-error";

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
 *
 * Pass `fromVoice` for a turn that came from speech, so the stored user row is
 * marked as a voice turn.
 */
export function sendMessage(
  text: string,
  onEvent: (event: ChatEvent) => void,
  fromVoice = false,
): Promise<void> {
  const channel = new Channel<ChatEvent>();
  channel.onmessage = onEvent;
  return invoke<void>("send_message", { text, fromVoice, onEvent: channel });
}

/** Calls `cb` for every engine state change. Remember to call the unlistener. */
export function onChatState(
  cb: (event: ChatStateEvent) => void,
): Promise<UnlistenFn> {
  return listen<ChatStateEvent>(CHAT_STATE_EVENT, (event) => {
    cb(event.payload);
  });
}

/** Reads the memory index's current state. */
export function getMemoryStatus(): Promise<MemoryStatus> {
  return invoke<MemoryStatus>("get_memory_status");
}

/** Calls `cb` for every memory state change. Remember to call the unlistener. */
export function onMemoryStatus(
  cb: (event: MemoryStatus) => void,
): Promise<UnlistenFn> {
  return listen<MemoryStatus>(MEMORY_STATUS_EVENT, (event) => {
    cb(event.payload);
  });
}

/** Reads the vision engine's current state. */
export function getVisionStatus(): Promise<VisionStatus> {
  return invoke<VisionStatus>("get_vision_status");
}

/** Calls `cb` for every vision state change. Remember to call the unlistener. */
export function onVisionStatus(
  cb: (event: VisionStatus) => void,
): Promise<UnlistenFn> {
  return listen<VisionStatus>(VISION_STATUS_EVENT, (event) => {
    cb(event.payload);
  });
}

/* ------------------------------------------------------------------ */
/* Voice commands                                                      */
/* ------------------------------------------------------------------ */

/** Reads the speech-to-text engine's current state. */
export function getVoiceStatus(): Promise<VoiceStatus> {
  return invoke<VoiceStatus>("get_voice_status");
}

/** Turns the microphone on or off. Errors reach the UI as a status event. */
export function setListening(on: boolean): Promise<void> {
  return invoke<void>("set_listening", { on });
}

/** Stops any spoken reply that is playing. */
export function stopSpeaking(): Promise<void> {
  return invoke<void>("stop_speaking");
}

/** Calls `cb` for every voice input state change. Remember to call the unlistener. */
export function onVoiceStatus(
  cb: (status: VoiceStatus) => void,
): Promise<UnlistenFn> {
  return listen<VoiceStatus>(VOICE_STATUS_EVENT, (event) => {
    cb(event.payload);
  });
}

/** Calls `cb` with the microphone's loudness. Remember to call the unlistener. */
export function onMicLevel(
  cb: (level: number) => void,
): Promise<UnlistenFn> {
  return listen<MicLevel>(MIC_LEVEL_EVENT, (event) => {
    cb(event.payload.level);
  });
}

/** Calls `cb` for every speech transcript. Remember to call the unlistener. */
export function onVoiceTranscript(
  cb: (text: string) => void,
): Promise<UnlistenFn> {
  return listen<VoiceTranscript>(VOICE_TRANSCRIPT_EVENT, (event) => {
    cb(event.payload.text);
  });
}

/** Calls `cb` when spoken output starts or stops. Remember to call the unlistener. */
export function onSpeaking(
  cb: (speaking: boolean) => void,
): Promise<UnlistenFn> {
  return listen<SpeakingPayload>(SPEAKING_EVENT, (event) => {
    cb(event.payload.speaking);
  });
}

/** Calls `cb` when spoken output fails. Remember to call the unlistener. */
export function onVoiceOutputError(
  cb: (message: string) => void,
): Promise<UnlistenFn> {
  return listen<VoiceOutputError>(VOICE_OUTPUT_ERROR_EVENT, (event) => {
    cb(event.payload.message);
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
