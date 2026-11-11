import { Mic, Search } from "lucide-react";

import type { ChatRole } from "../../lib/api.ts";

export interface MessageBubbleProps {
  role: ChatRole;
  text: string;
  fromVoice?: boolean;
  /** Shows the blinking caret after the text while a reply streams in. */
  streaming?: boolean;
  /** Muted text shown instead of the caret while `text` is still empty. */
  placeholder?: string;
  /** Tool labels shown as muted chips under the bubble. */
  tools?: string[];
}

/** One turn in the conversation: user on the right, assistant on the left. */
export function MessageBubble({
  role,
  text,
  fromVoice = false,
  streaming = false,
  placeholder,
  tools = [],
}: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={isUser ? "flex justify-end" : "flex justify-start"}>
      <div className="flex max-w-[85%] flex-col">
        <div
          className={[
            "rounded-xl px-3 py-2 text-sm break-words whitespace-pre-wrap",
            isUser
              ? "bg-accent-soft text-accent-soft-foreground"
              : "bg-surface-secondary text-surface-secondary-foreground",
          ].join(" ")}
        >
          {fromVoice ? (
            <Mic
              size={14}
              aria-label="Sent by voice"
              className="mr-1 inline-block align-text-bottom"
            />
          ) : null}
          {text.length > 0 || !placeholder ? (
            <>
              {text}
              {streaming ? (
                <span
                  aria-hidden="true"
                  className="ml-0.5 inline-block h-4 w-0.5 animate-caret-blink bg-current align-text-bottom"
                />
              ) : null}
            </>
          ) : (
            <span className="text-muted">{placeholder}</span>
          )}
        </div>
        {tools.length > 0 ? (
          <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted">
            {tools.map((label) => (
              <span key={label} className="inline-flex items-center gap-1">
                <Search size={12} aria-hidden="true" />
                {label}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
