import { describe, expect, it } from "vitest";
import { clamp, fileNameFromPath } from "./format.ts";

describe("fileNameFromPath", () => {
  it("reads the last segment of a Windows path", () => {
    expect(fileNameFromPath("C:\\models\\qwen\\model.gguf")).toBe("model.gguf");
  });

  it("reads the last segment of a POSIX path", () => {
    expect(fileNameFromPath("/home/me/models/model.gguf")).toBe("model.gguf");
  });

  it("handles mixed separators", () => {
    expect(fileNameFromPath("C:\\models/qwen\\model.gguf")).toBe("model.gguf");
  });

  it("ignores trailing separators", () => {
    expect(fileNameFromPath("/home/me/models/")).toBe("models");
    expect(fileNameFromPath("C:\\models\\")).toBe("models");
  });

  it("returns a bare file name unchanged", () => {
    expect(fileNameFromPath("model.gguf")).toBe("model.gguf");
  });

  it("returns an empty string for empty input", () => {
    expect(fileNameFromPath("")).toBe("");
    expect(fileNameFromPath("   ")).toBe("");
  });
});

describe("clamp", () => {
  it("returns values already in range", () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(0, 0, 10)).toBe(0);
    expect(clamp(10, 0, 10)).toBe(10);
  });

  it("raises values below the minimum", () => {
    expect(clamp(-3, 0, 10)).toBe(0);
  });

  it("lowers values above the maximum", () => {
    expect(clamp(42, 0, 10)).toBe(10);
  });

  it("falls back to the minimum for NaN", () => {
    expect(clamp(Number.NaN, 2, 10)).toBe(2);
  });
});
