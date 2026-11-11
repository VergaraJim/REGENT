import { useEffect, useRef } from "react";
import { Alert, Button, Spinner } from "@heroui/react";

import { Composer } from "../components/chat/Composer.tsx";
import { MessageBubble } from "../components/chat/MessageBubble.tsx";
import type { ChatController } from "../hooks/useChat.ts";
import type { MemoryStatus } from "../lib/api.ts";
import { toDisplayMessages } from "../lib/chatDisplay.ts";

export interface ChatViewProps {
  controller: ChatController;
  /** Whether the settings have a text model chosen. */
  textModelSet: boolean;
  onOpenSettings: () => void;
  /** The memory index's state, shown as a muted line under the composer. */
  memory?: MemoryStatus | null;
  /** A voice failure to show in red just above the memory line. */
  voiceNotice?: string | null;
}

/** The conversation, its states, and the composer pinned to the bottom. */
export function ChatView({
  controller,
  textModelSet,
  onOpenSettings,
  memory = null,
  voiceNotice = null,
}: ChatViewProps) {
  const {
    messages,
    chatState,
    error,
    turnError,
    streamingText,
    streamingTools,
    compactingReply,
  } = controller;
  const listRef = useRef<HTMLDivElement>(null);
  const nearBottom = useRef(true);

  useEffect(() => {
    const list = listRef.current;
    if (!list || !nearBottom.current) return;
    list.scrollTop = list.scrollHeight;
  }, [messages, streamingText, chatState]);

  const display = toDisplayMessages(messages);
  const memoryLine = describeMemory(memory);
  const idle = messages.length === 0 && streamingText === null;

  return (
    <main className="flex min-h-0 flex-1 flex-col">
      <div
        ref={listRef}
        role="log"
        aria-live="polite"
        aria-label="Conversation"
        onScroll={() => {
          const list = listRef.current;
          if (!list) return;
          nearBottom.current =
            list.scrollHeight - list.scrollTop - list.clientHeight < 48;
        }}
        className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-4"
      >
        {idle ? (
          <EmptyState
            chatState={chatState}
            textModelSet={textModelSet}
            onOpenSettings={onOpenSettings}
          />
        ) : (
          <>
            {display.map((message) => (
              <MessageBubble
                key={message.key}
                role={message.role}
                text={message.text}
                fromVoice={message.fromVoice}
                tools={message.tools}
              />
            ))}
            {streamingText !== null ? (
              <MessageBubble
                role="assistant"
                text={streamingText}
                streaming
                tools={streamingTools}
                placeholder={
                  compactingReply && streamingText === ""
                    ? "Compacting…"
                    : undefined
                }
              />
            ) : null}
          </>
        )}
      </div>

      <div className="flex shrink-0 flex-col gap-2 border-t border-separator p-4">
        {error ? (
          <Alert status="danger" className="text-sm">
            <Alert.Indicator />
            <Alert.Content>
              <Alert.Description className="break-words">
                {error}
              </Alert.Description>
              <div className="mt-2">
                <Button size="sm" variant="primary" onPress={controller.start}>
                  Start
                </Button>
              </div>
            </Alert.Content>
          </Alert>
        ) : null}
        {turnError ? (
          <Alert status="danger" className="text-sm">
            <Alert.Indicator />
            <Alert.Content>
              <Alert.Description className="break-words">
                {turnError}
              </Alert.Description>
            </Alert.Content>
          </Alert>
        ) : null}
        {chatState === "stopped" && messages.length > 0 ? (
          <p className="text-xs text-muted">
            Regent is stopped. Press Start to continue.
          </p>
        ) : null}
        {chatState === "compacting" ? (
          <div
            role="status"
            className="flex items-center gap-2 text-xs text-muted"
          >
            <Spinner size="sm" />
            <span>Compacting…</span>
          </div>
        ) : null}
        <Composer
          disabled={chatState !== "ready" && chatState !== "compacting"}
          generating={chatState === "generating"}
          onSend={controller.send}
          onCancel={controller.cancel}
        />
        {voiceNotice ? (
          <p
            role="status"
            aria-live="polite"
            title={voiceNotice}
            className="truncate text-xs text-danger"
          >
            {voiceNotice}
          </p>
        ) : null}
        {memoryLine ? (
          <p
            role="status"
            aria-live="polite"
            title={memoryLine}
            className="truncate text-xs text-muted"
          >
            {memoryLine}
          </p>
        ) : null}
      </div>
    </main>
  );
}

/** The muted footer line for `memory`, or `null` when there is nothing to say. */
function describeMemory(memory: MemoryStatus | null): string | null {
  if (memory === null) return null;
  switch (memory.state) {
    case "off":
      return memory.error;
    case "ready":
      return `Memory: ${memory.indexed} of ${memory.total} messages saved`;
    case "indexing":
      return `Memory: ${memory.indexed} of ${memory.total} messages saved · indexing…`;
    case "paused":
      return "Memory paused. Restart Regent to try again.";
  }
}

function EmptyState({
  chatState,
  textModelSet,
  onOpenSettings,
}: {
  chatState: ChatController["chatState"];
  textModelSet: boolean;
  onOpenSettings: () => void;
}) {
  if (chatState === "loading") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
        <Spinner size="sm" />
        <p className="text-sm text-muted">Loading model…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
      <p className="text-sm text-muted">Press Start to wake Regent.</p>
      {textModelSet ? null : (
        <>
          <p className="text-sm text-muted">Choose a model in Settings.</p>
          <Button
            size="sm"
            variant="ghost"
            className="underline"
            onPress={onOpenSettings}
          >
            Open Settings
          </Button>
        </>
      )}
    </div>
  );
}
