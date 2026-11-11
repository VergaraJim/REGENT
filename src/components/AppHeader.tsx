import { Button, Spinner } from "@heroui/react";
import { Settings, VolumeX } from "lucide-react";

import { MicButton } from "./MicButton.tsx";
import type { ChatController } from "../hooks/useChat.ts";
import type { VoiceController } from "../hooks/useVoice.ts";

export function AppHeader({
  chat,
  voice,
  sttModelSet,
  onOpenSettings,
}: {
  chat: ChatController;
  voice: VoiceController;
  sttModelSet: boolean;
  onOpenSettings: () => void;
}) {
  const { chatState, start, stop } = chat;
  const stopped = chatState === "stopped" || chatState === "error";

  return (
    <header className="flex h-12 shrink-0 items-center border-b border-separator px-4">
      <span className="text-sm font-semibold">Regent</span>
      <div className="flex grow items-center justify-end gap-2">
        <MicButton
          status={voice.status}
          level={voice.level}
          sttModelSet={sttModelSet}
          onToggle={voice.setListening}
          onOpenSettings={onOpenSettings}
        />
        {voice.speaking ? (
          <Button
            isIconOnly
            variant="ghost"
            size="sm"
            aria-label="Stop speaking"
            ref={(node) => {
              node?.setAttribute("title", "Stop speaking");
            }}
            onPress={voice.stopSpeaking}
          >
            <VolumeX size={18} aria-hidden="true" />
          </Button>
        ) : null}
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
