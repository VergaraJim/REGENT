import { describe, expect, it } from "vitest";
import type { ContextReport } from "./api.ts";
import { contextHint, statusColor, usagePercent } from "./memory.ts";

function report(
  reason: ContextReport["applied"]["reason"],
  limiting: ContextReport["limits"]["limiting"],
): ContextReport {
  return {
    limits: {
      model_max: 32768,
      system_max: 16384,
      effective_max: 16384,
      limiting,
      fits: true,
    },
    requested: 32768,
    applied: { value: 16384, reason },
  };
}

describe("statusColor", () => {
  it("maps ok to success", () => {
    expect(statusColor("ok")).toBe("success");
  });

  it("maps tight to warning", () => {
    expect(statusColor("tight")).toBe("warning");
  });

  it("maps too_big to danger", () => {
    expect(statusColor("too_big")).toBe("danger");
  });
});

describe("contextHint", () => {
  it("explains a reduction to the PC's maximum", () => {
    expect(contextHint(report("above_max", "this_pc"))).toBe(
      "Reduced to fit this PC",
    );
  });

  it("explains a reduction to the model's maximum", () => {
    expect(contextHint(report("above_max", "model"))).toBe(
      "Reduced to the model's maximum",
    );
  });

  it("explains a raise to the minimum", () => {
    expect(contextHint(report("raised_to_minimum", "unknown"))).toBe(
      "Raised to the minimum of 256",
    );
  });

  it("returns null for a context within its limits", () => {
    expect(contextHint(report("within_limits", "unknown"))).toBeNull();
    expect(contextHint(report("rounded", "equal"))).toBeNull();
  });

  it("returns null without a context report", () => {
    expect(contextHint(null)).toBeNull();
  });
});

describe("usagePercent", () => {
  it("computes a rounded percentage", () => {
    expect(usagePercent(5, 10)).toBe(50);
    expect(usagePercent(1, 3)).toBe(33);
  });

  it("caps the result at 100", () => {
    expect(usagePercent(12, 10)).toBe(100);
    expect(usagePercent(1000, 1)).toBe(100);
  });

  it("returns 0 when the budget is 0", () => {
    expect(usagePercent(5, 0)).toBe(0);
    expect(usagePercent(0, 0)).toBe(0);
  });
});
