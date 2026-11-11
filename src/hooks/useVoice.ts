import { useCallback, useEffect, useRef, useState } from "react";
import {
  getVoiceStatus,
  onMicLevel,
  onSpeaking,
  onVoiceOutputError,
  onVoiceStatus,
  onVoiceTranscript,
  setListening as setListeningCommand,
  stopSpeaking as stopSpeakingCommand,
} from "../lib/api.ts";
import type { VoiceStatus } from "../lib/api.ts";
import type { ChatController } from "./useChat.ts";

/** How many transcripts wait for the engine while a reply is still coming. */
const MAX_QUEUED = 3;

export interface VoiceController {
  /** The speech engine's state; `null` until the first read succeeds. */
  status: VoiceStatus | null;
  /** The microphone's latest loudness, 0.0 to 1.0; 0 when not listening. */
  level: number;
  /** Whether Regent is speaking a reply right now. */
  speaking: boolean;
  /** The last failure of the spoken output, shown under the composer. */
  outputError: string | null;
  /** Turns the microphone on or off. */
  setListening: (on: boolean) => void;
  /** Stops the spoken reply that is playing. */
  stopSpeaking: () => void;
}

/**
 * Owns voice input and output state for the whole app.
 *
 * Create it once (in `App`) so the microphone keeps its state across view
 * changes. Finished transcripts become voice turns: sent right away while the
 * engine is ready, queued while it is busy, and dropped otherwise.
 */
export function useVoice(chat: ChatController): VoiceController {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [level, setLevel] = useState(0);
  const [speaking, setSpeaking] = useState(false);
  const [outputError, setOutputError] = useState<string | null>(null);

  // The transcript listener is registered once, so it reads the chat through a
  // ref instead of capturing the first render's controller.
  const chatRef = useRef(chat);
  useEffect(() => {
    chatRef.current = chat;
  }, [chat]);

  const queued = useRef<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    const unlisteners: Array<() => void> = [];

    const track = (subscription: Promise<() => void>) => {
      void subscription.then((unlisten) => {
        if (cancelled) {
          unlisten();
        } else {
          unlisteners.push(unlisten);
        }
      });
    };

    getVoiceStatus()
      .then((next) => {
        if (!cancelled) setStatus(next);
      })
      .catch(() => {
        // No speech model is a normal state; the status stays null.
      });

    track(
      onVoiceStatus((next) => {
        if (!cancelled) setStatus(next);
      }),
    );

    track(
      onMicLevel((next) => {
        if (!cancelled) setLevel(next);
      }),
    );

    track(
      onVoiceTranscript((text) => {
        if (cancelled) return;
        const current = chatRef.current.chatState;
        if (current === "ready") {
          chatRef.current.send(text, true);
        } else if (current === "generating" || current === "compacting") {
          if (queued.current.length < MAX_QUEUED) queued.current.push(text);
        }
      }),
    );

    track(
      onSpeaking((next) => {
        if (cancelled) return;
        setSpeaking(next);
        if (next) setOutputError(null);
      }),
    );

    track(
      onVoiceOutputError((message) => {
        if (!cancelled) setOutputError(message);
      }),
    );

    return () => {
      cancelled = true;
      for (const unlisten of unlisteners) unlisten();
    };
  }, []);

  // A meter means nothing while the mic is idle or paused, so the reported
  // level is zero then even if a stale reading is still held.
  const listening = status !== null && status.listening && !status.paused;

  // Send the transcript that waited for the engine to become ready again.
  useEffect(() => {
    if (chat.chatState !== "ready") return;
    const next = queued.current.shift();
    if (next === undefined) return;
    chat.send(next, true);
  }, [chat]);

  const setListening = useCallback((on: boolean) => {
    setListeningCommand(on).catch(() => {
      // The failure reaches the UI through the status event.
    });
  }, []);

  const stopSpeaking = useCallback(() => {
    stopSpeakingCommand().catch(() => {
      // The failure reaches the UI through the status event.
    });
  }, []);

  return {
    status,
    level: listening ? level : 0,
    speaking,
    outputError,
    setListening,
    stopSpeaking,
  };
}
