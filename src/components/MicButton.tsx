import type { ReactNode } from "react";
import { Button, Spinner } from "@heroui/react";
import { Mic, MicOff } from "lucide-react";

import type { VoiceStatus } from "../lib/api.ts";

export interface MicButtonProps {
  /** The speech engine's state; `null` until the first read succeeds. */
  status: VoiceStatus | null;
  /** The microphone's latest loudness, 0.0 to 1.0. */
  level: number;
  /** Whether the settings have a speech model chosen. */
  sttModelSet: boolean;
  onToggle: (on: boolean) => void;
  onOpenSettings: () => void;
}

/** The header's microphone switch, with a small level meter beside it. */
export function MicButton({
  status,
  level,
  sttModelSet,
  onToggle,
  onOpenSettings,
}: MicButtonProps) {
  let icon: ReactNode;
  let label: string;
  let onPress: (() => void) | undefined;
  let disabled = false;
  let showMeter = false;

  if (!sttModelSet) {
    icon = <MicOff size={18} aria-hidden="true" className="text-muted" />;
    label = "Voice input needs a speech model. Open Settings.";
    onPress = onOpenSettings;
  } else if (status === null || status.state === "off") {
    icon = <MicOff size={18} aria-hidden="true" className="text-muted" />;
    label = "Press Start to use voice.";
    disabled = true;
  } else if (status.state === "loading") {
    icon = <Spinner size="sm" />;
    label = "Loading speech model…";
    disabled = true;
  } else if (status.state === "error") {
    icon = <MicOff size={18} aria-hidden="true" className="text-danger" />;
    label = status.error ?? "Voice input is unavailable.";
    onPress = onOpenSettings;
  } else if (status.listening && !status.paused) {
    icon = <Mic size={18} aria-hidden="true" className="text-accent" />;
    label = "Turn the microphone off";
    onPress = () => onToggle(false);
    showMeter = true;
  } else if (status.listening) {
    icon = <Mic size={18} aria-hidden="true" className="text-muted" />;
    label = "Microphone paused while Regent speaks";
    onPress = () => onToggle(false);
  } else {
    icon = <MicOff size={18} aria-hidden="true" />;
    label = "Turn the microphone on";
    onPress = () => onToggle(true);
  }

  return (
    <>
      <Button
        isIconOnly
        variant="ghost"
        size="sm"
        aria-label={label}
        ref={(node) => {
          // HeroUI's Button drops unknown props, so set the tooltip text here.
          node?.setAttribute("title", label);
        }}
        isDisabled={disabled}
        onPress={onPress}
      >
        {icon}
      </Button>
      {showMeter ? (
        <div
          data-testid="mic-level"
          aria-hidden="true"
          className="h-1 w-6 overflow-hidden rounded-full bg-separator"
        >
          <div
            className="h-full bg-accent transition-[width] duration-75"
            style={{ width: `${Math.min(1, level / 0.1) * 100}%` }}
          />
        </div>
      ) : null}
    </>
  );
}
