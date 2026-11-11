import { describe, expect, it } from "vitest";
import { toDisplayMessages, toolLabel } from "./chatDisplay.ts";
import type { ChatMessage } from "./api.ts";

function row(overrides: Partial<ChatMessage>): ChatMessage {
  return {
    id: 1,
    role: "assistant",
    text: "",
    from_voice: false,
    created_at: 1,
    ...overrides,
  };
}

describe("toolLabel", () => {
  it("names the known tools and falls back for anything else", () => {
    expect(toolLabel("search_memory")).toBe("Searched memory");
    expect(toolLabel("look_at_screen")).toBe("Looked at screen");
    expect(toolLabel("something_new")).toBe("Used a tool");
  });
});

describe("toDisplayMessages", () => {
  it("keeps a plain conversation as one bubble per row", () => {
    const display = toDisplayMessages([
      row({ id: 1, role: "user", text: "Hello" }),
      row({ id: 2, role: "assistant", text: "Hi" }),
      row({ id: 3, role: "user", text: "Again" }),
      row({ id: 4, role: "assistant", text: "Sure" }),
    ]);

    expect(display).toEqual([
      { key: 1, role: "user", text: "Hello", fromVoice: false, tools: [] },
      { key: 2, role: "assistant", text: "Hi", fromVoice: false, tools: [] },
      { key: 3, role: "user", text: "Again", fromVoice: false, tools: [] },
      { key: 4, role: "assistant", text: "Sure", fromVoice: false, tools: [] },
    ]);
  });

  it("merges an assistant row, a tool row and the continuation", () => {
    const display = toDisplayMessages([
      row({ id: 1, role: "user", text: "What did we decide?" }),
      row({ id: 2, role: "assistant", text: "Let me check." }),
      row({
        id: 3,
        role: "tool",
        text: "recalled text",
        tool_name: "search_memory",
        tool_call: "{}",
      }),
      row({ id: 4, role: "assistant", text: "We chose option B." }),
    ]);

    expect(display).toHaveLength(2);
    expect(display[1]).toEqual({
      key: 2,
      role: "assistant",
      text: "Let me check.\n\nWe chose option B.",
      fromVoice: false,
      tools: ["Searched memory"],
    });
  });

  it("starts the reply with the tool row when there is no lead-in text", () => {
    const display = toDisplayMessages([
      row({ id: 1, role: "user", text: "What did we decide?" }),
      row({ id: 2, role: "tool", text: "recalled", tool_name: "search_memory" }),
      row({ id: 3, role: "assistant", text: "Option B." }),
    ]);

    expect(display).toHaveLength(2);
    expect(display[1]).toEqual({
      key: 2,
      role: "assistant",
      text: "Option B.",
      fromVoice: false,
      tools: ["Searched memory"],
    });
  });

  it("shows a turn that ends on a tool row as an empty bubble with chips", () => {
    const display = toDisplayMessages([
      row({ id: 1, role: "user", text: "Look" }),
      row({ id: 2, role: "tool", text: "found", tool_name: "search_memory" }),
    ]);

    expect(display[1]).toEqual({
      key: 2,
      role: "assistant",
      text: "",
      fromVoice: false,
      tools: ["Searched memory"],
    });
  });

  it("keeps each tool label once, in order", () => {
    const display = toDisplayMessages([
      row({ id: 1, role: "user", text: "Look" }),
      row({ id: 2, role: "tool", tool_name: "search_memory" }),
      row({ id: 3, role: "tool", tool_name: "look_at_screen" }),
      row({ id: 4, role: "tool", tool_name: "search_memory" }),
      row({ id: 5, role: "assistant", text: "Done." }),
    ]);

    expect(display[1].tools).toEqual(["Searched memory", "Looked at screen"]);
  });

  it("labels an unknown tool generically", () => {
    const display = toDisplayMessages([
      row({ id: 1, role: "user", text: "Go" }),
      row({ id: 2, role: "tool", tool_name: "unknown" }),
    ]);

    expect(display[1].tools).toEqual(["Used a tool"]);
  });
});
