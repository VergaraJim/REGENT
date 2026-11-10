import type { Settings } from "./api.ts";

/** The last segment of a Windows or POSIX path. */
export function fileNameFromPath(path: string): string {
  const trimmed = path.trim();
  if (!trimmed) return "";
  const parts = trimmed.split(/[\\/]+/u).filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : trimmed;
}

/** Limits `n` to the inclusive range [min, max]. */
export function clamp(n: number, min: number, max: number): number {
  if (Number.isNaN(n)) return min;
  return Math.min(Math.max(n, min), max);
}

/** A plain-language message for anything thrown by an IPC call. */
export function errorMessage(error: unknown): string {
  if (typeof error === "string" && error.trim()) return error;
  if (error && typeof error === "object") {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) return message;
  }
  return "Regent couldn't complete that request.";
}

/** A stable signature of the settings the memory estimate depends on. */
export function estimateKey(settings: Settings | null): string {
  if (!settings) return "";
  const { models, memory } = settings;
  return JSON.stringify([
    models.text.path ?? null,
    models.text.gpu_layers,
    models.text.context_size,
    models.vision.model_path ?? null,
    models.vision.projector_path ?? null,
    models.vision.gpu_layers,
    models.embedding.path ?? null,
    models.embedding.gpu_layers,
    models.embedding.context_size,
    memory.ram_gb ?? null,
    memory.vram_gb ?? null,
  ]);
}
