import { Mic } from "lucide-react";

import type { ChatRole } from "../../lib/api.ts";

export interface MessageBubbleProps {
  role: ChatRole;
  text: string;
  fromVoice?: boolean;
  /** Shows the blinking caret after the text while a reply streams in. */
  streaming?: boolean;
}

/** One turn in the conversation: user on the right, assistant on the left. */
export function MessageBubble({
  role,
  text,
  fromVoice = false,
  streaming = false,
}: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={isUser ? "flex justify-end" : "flex justify-start"}>
      <div
        className={[
          "max-w-[85%] rounded-xl px-3 py-2 text-sm break-words whitespace-pre-wrap",
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
        {text}
        {streaming ? (
          <span
            aria-hidden="true"
            className="ml-0.5 inline-block h-4 w-0.5 animate-caret-blink bg-current align-text-bottom"
          />
        ) : null}
      </div>
    </div>
  );
}
