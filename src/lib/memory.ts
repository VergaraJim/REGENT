import type { ContextReport, FitStatus } from "./api.ts";

/** The colour accent used for a fit status. */
export type StatusColor = "success" | "warning" | "danger";

const STATUS_COLORS: Record<FitStatus, StatusColor> = {
  ok: "success",
  tight: "warning",
  too_big: "danger",
};

/** Maps a forecast's fit status to the only colour accent the card uses. */
export function statusColor(status: FitStatus): StatusColor {
  return STATUS_COLORS[status];
}

/**
 * Explains why the saved context size was changed, or `null` when it wasn't.
 */
export function contextHint(context: ContextReport | null): string | null {
  if (!context) return null;
  const { applied, limits } = context;
  if (applied.reason === "raised_to_minimum") {
    return "Raised to the minimum of 256";
  }
  if (applied.reason === "above_max") {
    if (limits.limiting === "this_pc") return "Reduced to fit this PC";
    if (limits.limiting === "model") return "Reduced to the model's maximum";
  }
  return null;
}

/** `used` as a percentage of `budget`, capped at 100; 0 when the budget is 0. */
export function usagePercent(used: number, budget: number): number {
  if (!(budget > 0)) return 0;
  const percent = (used / budget) * 100;
  return Math.min(100, Math.max(0, Math.round(percent)));
}
