import { Button, Spinner } from "@heroui/react";
import { Settings } from "lucide-react";

import type { ChatController } from "../hooks/useChat.ts";

export function AppHeader({
  chat,
  onOpenSettings,
}: {
  chat: ChatController;
  onOpenSettings: () => void;
}) {
  const { chatState, start, stop } = chat;
  const stopped = chatState === "stopped" || chatState === "error";

  return (
    <header className="flex h-12 shrink-0 items-center border-b border-separator px-4">
      <span className="text-sm font-semibold">Regent</span>
      <div className="flex grow items-center justify-end gap-2">
        {chatState === "loading" ? (
          <Button size="sm" variant="secondary" isDisabled>
            <Spinner size="sm" />
            Loading…
          </Button>
        ) : stopped ? (
          <Button size="sm" variant="primary" onPress={start}>
            Start
          </Button>
        ) : (
          <Button size="sm" variant="secondary" onPress={stop}>
            Stop
          </Button>
        )}
        <Button
          isIconOnly
          variant="ghost"
          size="sm"
          aria-label="Settings"
          onPress={onOpenSettings}
        >
          <Settings size={18} aria-hidden="true" />
        </Button>
      </div>
    </header>
  );
}
