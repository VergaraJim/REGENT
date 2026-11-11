// @vitest-environment jsdom
import { createElement } from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useChat } from "./useChat.ts";
import type { ChatController } from "./useChat.ts";
import type { ChatEvent, ChatMessage, ChatStateEvent } from "../lib/api.ts";

const api = vi.hoisted(() => ({
  getHistory: vi.fn(),
  getChatState: vi.fn(),
  onChatState: vi.fn(),
  startChat: vi.fn(),
  stopChat: vi.fn(),
  cancelGeneration: vi.fn(),
  sendMessage: vi.fn(),
}));

vi.mock("../lib/api.ts", () => api);

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

interface Harness {
  current: () => ChatController;
  unmount: () => void;
}

async function mount(
  state: ChatStateEvent,
  history: ChatMessage[] = [],
): Promise<Harness> {
  api.getHistory.mockResolvedValue(history);
  api.getChatState.mockResolvedValue(state);
  api.onChatState.mockResolvedValue(() => {});
  api.startChat.mockResolvedValue(undefined);
  api.stopChat.mockResolvedValue(undefined);
  api.cancelGeneration.mockResolvedValue(undefined);

  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latest: ChatController | null = null;

  function Probe() {
    latest = useChat();
    return null;
  }

  await act(async () => {
    root.render(createElement(Probe));
  });

  return {
    current: () => {
      if (!latest) throw new Error("hook has not rendered");
      return latest;
    },
    unmount: () => {
      act(() => root.unmount());
      container.remove();
    },
  };
}

let onEvent: ((event: ChatEvent) => void) | null = null;

beforeEach(() => {
  vi.clearAllMocks();
  onEvent = null;
  api.sendMessage.mockImplementation(
    (_text: string, callback: (event: ChatEvent) => void) => {
      onEvent = callback;
      return new Promise<void>(() => {});
    },
  );
});

afterEach(() => {
  document.body.innerHTML = "";
});

async function emit(event: ChatEvent) {
  if (!onEvent) throw new Error("send_message was not called");
  await act(async () => {
    onEvent?.(event);
  });
}

describe("useChat", () => {
  it("streams tokens into one in-progress assistant message", async () => {
    const chat = await mount({ state: "ready", error: null });

    await act(async () => {
      chat.current().send("Hello");
    });

    expect(api.sendMessage).toHaveBeenCalledWith("Hello", expect.any(Function));
    expect(chat.current().messages.map((message) => message.text)).toEqual([
      "Hello",
    ]);
    expect(chat.current().streamingText).toBe("");

    await emit({ type: "token", text: "Hel" });
    await emit({ type: "token", text: "lo" });
    expect(chat.current().streamingText).toBe("Hello");

    await emit({
      type: "done",
      message_id: 7,
      stats: { prompt_tokens: 1, generated_tokens: 2, tokens_per_second: 5 },
    });

    expect(chat.current().streamingText).toBeNull();
    expect(chat.current().messages.map((message) => message.text)).toEqual([
      "Hello",
      "Hello",
    ]);
    expect(chat.current().messages[1]).toMatchObject({
      id: 7,
      role: "assistant",
    });
    expect(chat.current().chatState).toBe("ready");
    chat.unmount();
  });

  it("keeps partial text when generation is cancelled", async () => {
    const chat = await mount({ state: "ready", error: null });

    await act(async () => {
      chat.current().send("Tell me");
    });
    await emit({ type: "token", text: "Par" });
    await emit({ type: "cancelled", message_id: 9 });

    expect(chat.current().streamingText).toBeNull();
    expect(chat.current().messages[1]).toMatchObject({
      id: 9,
      role: "assistant",
      text: "Par",
    });
    chat.unmount();
  });

  it("drops the in-progress reply when cancelling with no saved message", async () => {
    const chat = await mount({ state: "ready", error: null });

    await act(async () => {
      chat.current().send("Tell me");
    });
    await emit({ type: "cancelled", message_id: null });

    expect(chat.current().streamingText).toBeNull();
    expect(chat.current().messages).toHaveLength(1);
    chat.unmount();
  });

  it("shows the backend error and stops streaming", async () => {
    const chat = await mount({ state: "ready", error: null });

    await act(async () => {
      chat.current().send("Hello");
    });
    await emit({ type: "token", text: "Hi" });
    await emit({ type: "error", message: "The model ran out of memory." });

    expect(chat.current().streamingText).toBeNull();
    expect(chat.current().error).toBe("The model ran out of memory.");
    expect(chat.current().chatState).toBe("error");
    chat.unmount();
  });

  it("loads history on mount and ignores sends until ready", async () => {
    const chat = await mount({ state: "stopped", error: null }, [
      {
        id: 1,
        role: "user",
        text: "Earlier",
        from_voice: true,
        created_at: 1,
      },
    ]);

    expect(chat.current().messages).toHaveLength(1);
    await act(async () => {
      chat.current().send("Too early");
    });
    expect(api.sendMessage).not.toHaveBeenCalled();
    chat.unmount();
  });
});
