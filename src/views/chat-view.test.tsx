import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ChatView } from "./ChatView.tsx";
import type { ChatController } from "../hooks/useChat.ts";
import type { ChatMessage, ChatState } from "../lib/api.ts";

function controller(
  chatState: ChatState,
  overrides: Partial<ChatController> = {},
): ChatController {
  return {
    messages: [],
    chatState,
    error: null,
    turnError: null,
    streamingText: null,
    start: () => {},
    stop: () => {},
    send: () => {},
    cancel: () => {},
    ...overrides,
  };
}

function render(chat: ChatController, textModelSet = true): string {
  return renderToStaticMarkup(
    <ChatView
      controller={chat}
      textModelSet={textModelSet}
      onOpenSettings={() => {}}
    />,
  );
}

const messages: ChatMessage[] = [
  {
    id: 1,
    role: "user",
    text: "Hello there",
    from_voice: true,
    created_at: 1,
  },
  {
    id: 2,
    role: "assistant",
    text: "Hi! How can I help?",
    from_voice: false,
    created_at: 2,
  },
];

describe("ChatView markup", () => {
  it("shows the empty hint while stopped", () => {
    const html = render(controller("stopped"));
    expect(html).toContain("Press Start to wake Regent.");
    expect(html).toContain('role="log"');
  });

  it("asks for a model when none is chosen", () => {
    const html = render(controller("stopped"), false);
    expect(html).toContain("Choose a model in Settings.");
    expect(html).toContain("Open Settings");
  });

  it("shows the loading state", () => {
    const html = render(controller("loading"));
    expect(html).toContain("Loading model…");
  });

  it("renders the conversation and the error banner", () => {
    const html = render(
      controller("ready", {
        messages,
        error: "The model ran out of memory.",
      }),
    );
    expect(html).toContain("Hello there");
    expect(html).toContain("Hi! How can I help?");
    expect(html).toContain("Sent by voice");
    expect(html).toContain("The model ran out of memory.");
    expect(html).toContain('aria-label="Send"');
  });

  it("shows a turn error without a Start button", () => {
    const html = render(
      controller("ready", {
        messages,
        turnError: "The model ran out of memory.",
      }),
    );
    expect(html).toContain("The model ran out of memory.");
    expect(html).not.toContain("Start");
  });

  it("tells the user how to continue when stopped with a conversation", () => {
    const html = render(controller("stopped", { messages }));
    expect(html).toContain("Regent is stopped. Press Start to continue.");
    expect(html).toContain("Hello there");
  });

  it("turns Send into Cancel while generating", () => {
    const html = render(
      controller("generating", { messages, streamingText: "Partial" }),
    );
    expect(html).toContain('aria-label="Cancel"');
    expect(html).toContain("Partial");
  });
});
