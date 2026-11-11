import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ChatView } from "./ChatView.tsx";
import type { ChatController } from "../hooks/useChat.ts";
import type { ChatMessage, ChatState, MemoryStatus } from "../lib/api.ts";

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
    streamingTools: [],
    compactingReply: false,
    start: () => {},
    stop: () => {},
    send: () => {},
    cancel: () => {},
    ...overrides,
  };
}

function render(
  chat: ChatController,
  textModelSet = true,
  memory: MemoryStatus | null = null,
): string {
  return renderToStaticMarkup(
    <ChatView
      controller={chat}
      textModelSet={textModelSet}
      onOpenSettings={() => {}}
      memory={memory}
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

  it("shows idle compaction and keeps the composer enabled", () => {
    const html = render(controller("compacting", { messages }));
    expect(html).toContain('role="status"');
    expect(html).toContain("Compacting…");
    expect(html).toContain('aria-label="Send"');

    const textarea = /<textarea[^>]*aria-label="Message"[^>]*>/.exec(html)?.[0];
    expect(textarea).toBeTruthy();
    expect(textarea).not.toContain("disabled");
  });

  it("shows compaction inside the reply bubble before the first token", () => {
    const html = render(
      controller("ready", {
        messages,
        streamingText: "",
        compactingReply: true,
      }),
    );
    expect(html.match(/Compacting…/g)).toHaveLength(1);

    const log = html.indexOf('role="log"');
    const hint = html.indexOf("Compacting…");
    const composer = html.indexOf('aria-label="Message"');
    expect(log).toBeGreaterThanOrEqual(0);
    expect(hint).toBeGreaterThan(log);
    expect(hint).toBeLessThan(composer);
  });

  it("shows a reply that used a tool as one bubble with a chip", () => {
    const html = render(
      controller("ready", {
        messages: [
          ...messages,
          {
            id: 3,
            role: "tool",
            text: "recalled text",
            from_voice: false,
            created_at: 3,
            tool_name: "search_memory",
            tool_call: "{}",
          },
          {
            id: 4,
            role: "assistant",
            text: "It was option B.",
            from_voice: false,
            created_at: 4,
          },
        ],
      }),
    );

    expect(html.match(/Searched memory/g)).toHaveLength(1);
    expect(html.match(/Hi! How can I help\?/g)).toHaveLength(1);
    expect(html).toContain("It was option B.");
    expect(html).not.toContain("recalled text");
  });

  it("shows the memory count while ready", () => {
    const html = render(controller("ready", { messages }), true, {
      state: "ready",
      indexed: 3,
      total: 5,
      error: null,
    });
    expect(html).toContain("Memory: 3 of 5 messages saved");
    expect(html).toContain('aria-live="polite"');
    expect(html).not.toContain("indexing…");
  });

  it("shows indexing after the count", () => {
    const html = render(controller("ready", { messages }), true, {
      state: "indexing",
      indexed: 1,
      total: 5,
      error: null,
    });
    expect(html).toContain("Memory: 1 of 5 messages saved · indexing…");
  });

  it("shows the paused hint", () => {
    const html = render(controller("ready", { messages }), true, {
      state: "paused",
      indexed: 0,
      total: 0,
      error: null,
    });
    expect(html).toContain("Memory paused. Restart Regent to try again.");
  });

  it("renders no memory line without a memory model", () => {
    const html = render(controller("ready", { messages }), true, null);
    expect(html).not.toContain("Memory");
    expect(html).not.toContain("indexing…");

    const off = render(controller("ready", { messages }), true, {
      state: "off",
      indexed: 0,
      total: 0,
      error: null,
    });
    expect(off).not.toContain("Memory");
  });

  it("shows a memory error while off", () => {
    const html = render(controller("ready", { messages }), true, {
      state: "off",
      indexed: 0,
      total: 0,
      error: "The embedding model could not load.",
    });
    expect(html).toContain("The embedding model could not load.");
    expect(html).toContain("truncate");
  });
});
