// @vitest-environment jsdom
import { createElement } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MicButton } from "./MicButton.tsx";
import type { VoiceStatus } from "../lib/api.ts";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const ready: VoiceStatus = {
  state: "ready",
  listening: false,
  paused: false,
  error: null,
};

interface Harness {
  button: () => HTMLButtonElement;
  meter: () => HTMLElement | null;
  unmount: () => void;
}

async function mount(props: {
  status: VoiceStatus | null;
  sttModelSet?: boolean;
  level?: number;
  onToggle?: (on: boolean) => void;
  onOpenSettings?: () => void;
}): Promise<Harness> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  await act(async () => {
    root.render(
      createElement(MicButton, {
        status: props.status,
        sttModelSet: props.sttModelSet ?? true,
        level: props.level ?? 0,
        onToggle: props.onToggle ?? (() => {}),
        onOpenSettings: props.onOpenSettings ?? (() => {}),
      }),
    );
  });

  return {
    button: () => {
      const button = container.querySelector("button");
      if (!button) throw new Error("no button rendered");
      return button;
    },
    meter: () => container.querySelector('[data-testid="mic-level"]'),
    unmount: () => {
      act(() => root.unmount());
      container.remove();
    },
  };
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("MicButton", () => {
  it("asks for a speech model when none is chosen", async () => {
    const view = await mount({ status: ready, sttModelSet: false });
    expect(view.button().getAttribute("aria-label")).toBe(
      "Voice input needs a speech model. Open Settings.",
    );
    expect(view.button().getAttribute("title")).toBe(
      "Voice input needs a speech model. Open Settings.",
    );
    view.unmount();
  });

  it("waits for Start while the status is unknown", async () => {
    const view = await mount({ status: null });
    expect(view.button().getAttribute("aria-label")).toBe(
      "Press Start to use voice.",
    );
    expect(view.button().disabled).toBe(true);
    view.unmount();
  });

  it("waits for Start while the voice engine is off", async () => {
    const view = await mount({
      status: { state: "off", listening: false, paused: false, error: null },
    });
    expect(view.button().getAttribute("aria-label")).toBe(
      "Press Start to use voice.",
    );
    expect(view.button().disabled).toBe(true);
    view.unmount();
  });

  it("says the speech model is loading", async () => {
    const view = await mount({
      status: { state: "loading", listening: false, paused: false, error: null },
    });
    expect(view.button().getAttribute("aria-label")).toBe(
      "Loading speech model…",
    );
    expect(view.button().disabled).toBe(true);
    view.unmount();
  });

  it("shows the status error", async () => {
    const onOpenSettings = vi.fn();
    const view = await mount({
      status: {
        state: "error",
        listening: false,
        paused: false,
        error: "The microphone could not start.",
      },
      onOpenSettings,
    });
    expect(view.button().getAttribute("aria-label")).toBe(
      "The microphone could not start.",
    );
    await act(async () => {
      view.button().click();
    });
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
    view.unmount();
  });

  it("labels a live microphone and turns it off", async () => {
    const onToggle = vi.fn();
    const view = await mount({
      status: { state: "ready", listening: true, paused: false, error: null },
      onToggle,
    });
    expect(view.button().getAttribute("aria-label")).toBe(
      "Turn the microphone off",
    );
    await act(async () => {
      view.button().click();
    });
    expect(onToggle).toHaveBeenCalledWith(false);
    view.unmount();
  });

  it("labels a mic paused while Regent speaks", async () => {
    const onToggle = vi.fn();
    const view = await mount({
      status: { state: "ready", listening: true, paused: true, error: null },
      onToggle,
    });
    expect(view.button().getAttribute("aria-label")).toBe(
      "Microphone paused while Regent speaks",
    );
    await act(async () => {
      view.button().click();
    });
    expect(onToggle).toHaveBeenCalledWith(false);
    view.unmount();
  });

  it("turns the microphone on", async () => {
    const onToggle = vi.fn();
    const view = await mount({ status: ready, onToggle });
    expect(view.button().getAttribute("aria-label")).toBe(
      "Turn the microphone on",
    );
    await act(async () => {
      view.button().click();
    });
    expect(onToggle).toHaveBeenCalledWith(true);
    view.unmount();
  });

  it("opens Settings when the model is missing", async () => {
    const onOpenSettings = vi.fn();
    const view = await mount({
      status: ready,
      sttModelSet: false,
      onOpenSettings,
    });
    await act(async () => {
      view.button().click();
    });
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
    view.unmount();
  });

  it("shows the meter only while listening and not paused", async () => {
    const live = await mount({
      status: { state: "ready", listening: true, paused: false, error: null },
      level: 0.05,
    });
    const meter = live.meter();
    expect(meter).not.toBeNull();
    expect(meter?.getAttribute("aria-hidden")).toBe("true");
    expect(meter?.innerHTML).toContain("width: 50%");
    live.unmount();

    for (const status of [
      ready,
      { state: "ready", listening: true, paused: true, error: null },
      { state: "off", listening: false, paused: false, error: null },
      { state: "loading", listening: false, paused: false, error: null },
      { state: "error", listening: false, paused: false, error: null },
    ] satisfies VoiceStatus[]) {
      const view = await mount({ status, level: 0.05 });
      expect(view.meter()).toBeNull();
      view.unmount();
    }
  });
});
