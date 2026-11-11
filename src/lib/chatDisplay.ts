/**
 * Turns stored chat rows into the bubbles the conversation shows.
 *
 * A reply that used a tool is stored as an optional assistant row, one or more
 * `tool` rows, then the assistant continuation. The UI collapses that back into
 * a single assistant bubble with one chip per tool under it.
 */
import type { ChatMessage } from "./api.ts";

/** The chip text for one tool name. */
export function toolLabel(name: string): string {
  switch (name) {
    case "search_memory":
      return "Searched memory";
    case "look_at_screen":
      return "Looked at screen";
    default:
      return "Used a tool";
  }
}

/** One bubble in the conversation, tool rows folded into its chips. */
export interface DisplayMessage {
  /** The id of the first row this bubble was built from. */
  key: number;
  role: "user" | "assistant";
  text: string;
  fromVoice: boolean;
  /** The unique tool labels of the turn, in order. */
  tools: string[];
}

/** Folds tool rows into the assistant reply of the same turn. */
export function toDisplayMessages(messages: ChatMessage[]): DisplayMessage[] {
  const items: DisplayMessage[] = [];
  let current: DisplayMessage | null = null;

  for (const message of messages) {
    if (message.role === "tool") {
      if (current === null) {
        current = {
          key: message.id,
          role: "assistant",
          text: "",
          fromVoice: false,
          tools: [],
        };
        items.push(current);
      }
      const label = toolLabel(message.tool_name ?? "");
      if (!current.tools.includes(label)) current.tools.push(label);
      continue;
    }

    if (message.role === "user") {
      current = null;
      items.push({
        key: message.id,
        role: "user",
        text: message.text,
        fromVoice: message.from_voice,
        tools: [],
      });
      continue;
    }

    if (current === null) {
      current = {
        key: message.id,
        role: "assistant",
        text: "",
        fromVoice: message.from_voice,
        tools: [],
      };
      items.push(current);
    } else if (message.from_voice) {
      current.fromVoice = true;
    }
    if (message.text.length > 0) {
      current.text =
        current.text.length > 0 ? `${current.text}\n\n${message.text}` : message.text;
    }
  }

  return items;
}
