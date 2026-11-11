// @vitest-environment jsdom
import { createElement } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useVisionStatus } from "./useVisionStatus.ts";
import type { VisionStatus } from "../lib/api.ts";

const api = vi.hoisted(() => ({
  getVisionStatus: vi.fn(),
  onVisionStatus: vi.fn(),
}));

vi.mock("../lib/api.ts", () => api);

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let status: ((next: VisionStatus) => void) | null = null;

interface Harness {
  current: () => VisionStatus | null;
  unmount: () => void;
}

async function mount(): Promise<Harness> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latest: VisionStatus | null = null;

  function Probe() {
    latest = useVisionStatus();
    return null;
  }

  await act(async () => {
    root.render(createElement(Probe));
  });

  return {
    current: () => latest,
    unmount: () => {
      act(() => root.unmount());
      container.remove();
    },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  status = null;

  api.getVisionStatus.mockResolvedValue({ state: "off", error: null });
  api.onVisionStatus.mockImplementation((cb: (next: VisionStatus) => void) => {
    status = cb;
    return Promise.resolve(() => {});
  });
});

afterEach(() => {
  document.body.innerHTML = "";
});

describe("useVisionStatus", () => {
  it("loads the current status once on mount", async () => {
    api.getVisionStatus.mockResolvedValue({ state: "ready", error: null });
    const vision = await mount();

    expect(api.getVisionStatus).toHaveBeenCalledTimes(1);
    expect(api.onVisionStatus).toHaveBeenCalledTimes(1);
    expect(vision.current()).toEqual({ state: "ready", error: null });
    vision.unmount();
  });

  it("follows status events after the initial read", async () => {
    const vision = await mount();

    await act(async () => {
      status?.({ state: "loading", error: null });
    });
    expect(vision.current()).toEqual({ state: "loading", error: null });

    await act(async () => {
      status?.({ state: "idle", error: "The projector could not load." });
    });
    expect(vision.current()).toEqual({
      state: "idle",
      error: "The projector could not load.",
    });
    vision.unmount();
  });

  it("stays null when the initial read fails", async () => {
    api.getVisionStatus.mockRejectedValue(new Error("no vision"));
    const vision = await mount();

    expect(vision.current()).toBeNull();
    vision.unmount();
  });

  it("unlistens on unmount", async () => {
    const stop = vi.fn();
    api.onVisionStatus.mockResolvedValue(stop);
    const vision = await mount();

    vision.unmount();
    expect(stop).toHaveBeenCalledTimes(1);
  });
});
