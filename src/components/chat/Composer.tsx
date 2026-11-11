import { useState } from "react";
import { Button, TextArea } from "@heroui/react";

export interface ComposerProps {
  /** Whether the engine can accept a message right now. */
  disabled: boolean;
  /** Whether a reply is streaming in, which turns Send into Cancel. */
  generating: boolean;
  onSend: (text: string) => void;
  onCancel: () => void;
}

/** The chat input pinned to the bottom of the screen. */
export function Composer({
  disabled,
  generating,
  onSend,
  onCancel,
}: ComposerProps) {
  const [text, setText] = useState("");

  function submit() {
    const trimmed = text.trim();
    if (!trimmed) return;
    setText("");
    onSend(trimmed);
  }

  return (
    <div className="flex items-end gap-2">
      <TextArea
        aria-label="Message"
        rows={1}
        placeholder="Message Regent…"
        value={text}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== "Enter" || event.shiftKey) return;
          event.preventDefault();
          submit();
        }}
        className="min-h-0 max-h-32 w-full resize-none text-sm"
      />
      {generating ? (
        <Button
          variant="secondary"
          size="sm"
          aria-label="Cancel"
          onPress={onCancel}
        >
          Cancel
        </Button>
      ) : (
        <Button
          variant="primary"
          size="sm"
          aria-label="Send"
          isDisabled={disabled || !text.trim()}
          onPress={submit}
        >
          Send
        </Button>
      )}
    </div>
  );
}
