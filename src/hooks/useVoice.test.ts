// @vitest-environment jsdom
import { createElement } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useVoice } from "./useVoice.ts";
import type { VoiceController } from "./useVoice.ts";
import type { ChatController } from "./useChat.ts";
import type { ChatState } from "../lib/api.ts";

const api = vi.hoisted(() => ({
  getVoiceStatus: vi.fn(),
  onVoiceStatus: vi.fn(),
  onMicLevel: vi.fn(),
  onVoiceTranscript: vi.fn(),
  onSpeaking: vi.fn(),
  onVoiceOutputError: vi.fn(),
  setListening: vi.fn(),
  stopSpeaking: vi.fn(),
}));

vi.mock("../lib/api.ts", () => api);

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let transcript: ((text: string) => void) | null = null;
let speaking: ((next: boolean) => void) | null = null;
let outputError: ((message: string) => void) | null = null;

interface Harness {
  current: () => VoiceController;
  update: (chat: ChatController) => Promise<void>;
  unmount: () => void;
}

function fakeChat(
  chatState: ChatState,
  send: ChatController["send"],
): ChatController {
  return {
    messages: [],
    chatState,
    error: null,
    turnError: null,
    streamingText: null,
    streamingTools: [],
    compactingReply: false,
    start: () => {},
    stop: () => {},
    send,
    cancel: () => {},
  };
}

async function mount(chat: ChatController): Promise<Harness> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latest: VoiceController | null = null;

  function Probe({ current }: { current: ChatController }) {
    latest = useVoice(current);
    return null;
  }

  await act(async () => {
    root.render(createElement(Probe, { current: chat }));
  });

  return {
    current: () => {
      if (!latest) throw new Error("hook has not rendered");
      return latest;
    },
    update: async (next: ChatController) => {
      await act(async () => {
        root.render(createElement(Probe, { current: next }));
      });
    },
    unmount: () => {
      act(() => root.unmount());
      container.remove();
    },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  transcript = null;
  speaking = null;
  outputError = null;

  api.getVoiceStatus.mockResolvedValue({
    state: "off",
    listening: false,
    paused: false,
    error: null,
  });
  api.onVoiceStatus.mockResolvedValue(() => {});
  api.onMicLevel.mockResolvedValue(() => {});
  api.onVoiceTranscript.mockImplementation(
    (cb: (text: string) => void) => {
      transcript = cb;
      return Promise.resolve(() => {});
    },
  );
  api.onSpeaking.mockImplementation((cb: (next: boolean) => void) => {
    speaking = cb;
    return Promise.resolve(() => {});
  });
  api.onVoiceOutputError.mockImplementation(
    (cb: (message: string) => void) => {
      outputError = cb;
      return Promise.resolve(() => {});
    },
  );
  api.setListening.mockResolvedValue(undefined);
  api.stopSpeaking.mockResolvedValue(undefined);
});

afterEach(() => {
  document.body.innerHTML = "";
});

async function emitTranscript(text: string) {
  if (!transcript) throw new Error("transcript listener was not registered");
  await act(async () => {
    transcript?.(text);
  });
}

describe("useVoice", () => {
  it("sends a transcript while the engine is ready", async () => {
    const send = vi.fn();
    const voice = await mount(fakeChat("ready", send));

    await emitTranscript("hello");

    expect(send).toHaveBeenCalledWith("hello", true);
    voice.unmount();
  });

  it("sends a queued transcript once the engine is ready", async () => {
    const send = vi.fn();
    const voice = await mount(fakeChat("generating", send));

    await emitTranscript("hello");
    expect(send).not.toHaveBeenCalled();

    await voice.update(fakeChat("ready", send));
    expect(send).toHaveBeenCalledWith("hello", true);
    voice.unmount();
  });

  it("drops a transcript while the engine is stopped", async () => {
    const send = vi.fn();
    const voice = await mount(fakeChat("stopped", send));

    await emitTranscript("hello");
    expect(send).not.toHaveBeenCalled();

    await voice.update(fakeChat("ready", send));
    expect(send).not.toHaveBeenCalled();
    voice.unmount();
  });

  it("tracks speaking from the backend", async () => {
    const voice = await mount(fakeChat("ready", vi.fn()));
    expect(voice.current().speaking).toBe(false);

    await act(async () => {
      speaking?.(true);
    });
    expect(voice.current().speaking).toBe(true);

    await act(async () => {
      speaking?.(false);
    });
    expect(voice.current().speaking).toBe(false);
    voice.unmount();
  });

  it("clears an output error once Regent starts speaking", async () => {
    const voice = await mount(fakeChat("ready", vi.fn()));

    await act(async () => {
      outputError?.("The voice could not play.");
    });
    expect(voice.current().outputError).toBe("The voice could not play.");

    await act(async () => {
      speaking?.(true);
    });
    expect(voice.current().outputError).toBeNull();
    voice.unmount();
  });

  it("swallows command failures so the status event reports them", async () => {
    api.setListening.mockRejectedValue(new Error("no mic"));
    api.stopSpeaking.mockRejectedValue(new Error("no speaker"));
    const voice = await mount(fakeChat("ready", vi.fn()));

    await act(async () => {
      voice.current().setListening(true);
      voice.current().stopSpeaking();
    });

    expect(api.setListening).toHaveBeenCalledWith(true);
    expect(api.stopSpeaking).toHaveBeenCalledTimes(1);
    voice.unmount();
  });
});
