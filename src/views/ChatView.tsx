import { useEffect, useRef } from "react";
import { Alert, Button, Spinner } from "@heroui/react";

import { Composer } from "../components/chat/Composer.tsx";
import { MessageBubble } from "../components/chat/MessageBubble.tsx";
import type { ChatController } from "../hooks/useChat.ts";

export interface ChatViewProps {
  controller: ChatController;
  /** Whether the settings have a text model chosen. */
  textModelSet: boolean;
  onOpenSettings: () => void;
}

/** The conversation, its states, and the composer pinned to the bottom. */
export function ChatView({
  controller,
  textModelSet,
  onOpenSettings,
}: ChatViewProps) {
  const { messages, chatState, error, turnError, streamingText } = controller;
  const listRef = useRef<HTMLDivElement>(null);
  const nearBottom = useRef(true);

  useEffect(() => {
    const list = listRef.current;
    if (!list || !nearBottom.current) return;
    list.scrollTop = list.scrollHeight;
  }, [messages, streamingText, chatState]);

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
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                role={message.role}
                text={message.text}
                fromVoice={message.from_voice}
              />
            ))}
            {streamingText !== null ? (
              <MessageBubble role="assistant" text={streamingText} streaming />
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
        <Composer
          disabled={chatState !== "ready"}
          generating={chatState === "generating"}
          onSend={controller.send}
          onCancel={controller.cancel}
        />
      </div>
    </main>
  );
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
