import { describe, expect, it } from "vitest";
import { createLatestOnly } from "./api.ts";

describe("createLatestOnly", () => {
  it("accepts any generation before anything is tracked", () => {
    const latest = createLatestOnly();
    expect(latest.accept(0)).toBe(true);
    expect(latest.accept(1)).toBe(true);
  });

  it("accepts the newest tracked generation", () => {
    const latest = createLatestOnly();
    latest.track(3);
    expect(latest.accept(3)).toBe(true);
  });

  it("rejects a generation older than the newest tracked one", () => {
    const latest = createLatestOnly();
    latest.track(3);
    expect(latest.accept(2)).toBe(false);
    expect(latest.accept(1)).toBe(false);
  });

  it("never goes backwards when an older generation is tracked late", () => {
    const latest = createLatestOnly();
    latest.track(5);
    latest.track(4);
    expect(latest.accept(4)).toBe(false);
    expect(latest.accept(5)).toBe(true);
  });

  it("accepts generations newer than the newest tracked one", () => {
    const latest = createLatestOnly();
    latest.track(2);
    expect(latest.accept(3)).toBe(true);
  });
});
