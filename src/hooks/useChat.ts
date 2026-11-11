import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelGeneration,
  getChatState,
  getHistory,
  onChatState,
  sendMessage,
  startChat,
  stopChat,
} from "../lib/api.ts";
import type { ChatMessage, ChatState } from "../lib/api.ts";
import { toolLabel } from "../lib/chatDisplay.ts";
import { errorMessage } from "../lib/format.ts";

export interface ChatController {
  /** The stored conversation, oldest first. */
  messages: ChatMessage[];
  chatState: ChatState;
  /** The last start-up or engine-state failure, if any. */
  error: string | null;
  /** The last chat turn's failure, shown until the next send, Start or Stop. */
  turnError: string | null;
  /** Text of the assistant reply still streaming in, or `null` when idle. */
  streamingText: string | null;
  /** Tool labels for the reply still streaming in, in order. */
  streamingTools: string[];
  /** Whether the current reply is waiting for old messages to be summarized. */
  compactingReply: boolean;
  /** Loads the model and gets the engine ready. */
  start: () => void;
  /** Unloads the engine; the conversation stays on screen. */
  stop: () => void;
  /** Streams one reply to `text`. Does nothing unless the engine is ready. */
  send: (text: string, fromVoice?: boolean) => void;
  /** Asks the engine to stop generating, keeping any partial reply. */
  cancel: () => void;
}

/**
 * Owns the chat engine's state for the whole app.
 *
 * Create it once (in `App`) so switching to Settings and back does not lose
 * the conversation. History is loaded on mount and engine state is kept in
 * sync through the `chat-state` event.
 */
export function useChat(): ChatController {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatState, setChatState] = useState<ChatState>("stopped");
  const [error, setError] = useState<string | null>(null);
  const [turnError, setTurnError] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const [streamingTools, setStreamingTools] = useState<string[]>([]);
  const [compactingReply, setCompactingReply] = useState(false);

  const alive = useRef(true);
  const stateRef = useRef<ChatState>("stopped");
  const streaming = useRef<string | null>(null);
  const nextLocalId = useRef(-1);
  const turnUsedTool = useRef(false);

  const applyState = useCallback((state: ChatState) => {
    stateRef.current = state;
    setChatState(state);
  }, []);

  useEffect(() => {
    alive.current = true;
    let cancelled = false;
    let unlisten: (() => void) | undefined;

    getHistory()
      .then((history) => {
        if (!cancelled) setMessages(history);
      })
      .catch((failure: unknown) => {
        if (!cancelled) setError(errorMessage(failure));
      });

    getChatState()
      .then((current) => {
        if (cancelled) return;
        applyState(current.state);
        setError(current.error);
      })
      .catch((failure: unknown) => {
        if (!cancelled) setError(errorMessage(failure));
      });

    void onChatState((next) => {
      if (!alive.current) return;
      applyState(next.state);
      setError(next.error);
    }).then((stop) => {
      if (cancelled) {
        stop();
      } else {
        unlisten = stop;
      }
    });

    return () => {
      cancelled = true;
      alive.current = false;
      unlisten?.();
    };
  }, [applyState]);

  const finishStream = useCallback(
    (messageId: number | null) => {
      const text = streaming.current ?? "";
      const usedTool = turnUsedTool.current;
      turnUsedTool.current = false;
      streaming.current = null;
      setStreamingText(null);
      setStreamingTools([]);
      setCompactingReply(false);

      if (usedTool) {
        // The backend stored the turn as assistant row(s) plus tool row(s), so
        // the local append would show the wrong shape. Reload instead; the
        // optimistic user message stays until the reload resolves.
        getHistory()
          .then((history) => {
            if (!alive.current) return;
            setMessages(history);
          })
          .catch((failure: unknown) => {
            if (!alive.current) return;
            setTurnError(errorMessage(failure));
            if (messageId !== null && text.length > 0) {
              setMessages((previous) => [
                ...previous,
                {
                  id: messageId,
                  role: "assistant",
                  text,
                  from_voice: false,
                  created_at: Date.now(),
                  tool_name: null,
                  tool_call: null,
                },
              ]);
            }
          });
      } else if (messageId !== null && text.length > 0) {
        setMessages((previous) => [
          ...previous,
          {
            id: messageId,
            role: "assistant",
            text,
            from_voice: false,
            created_at: Date.now(),
          },
        ]);
      }
      if (stateRef.current === "generating") applyState("ready");
    },
    [applyState],
  );

  const send = useCallback(
    (raw: string, fromVoice = false) => {
      const text = raw.trim();
      if (
        !text ||
        (stateRef.current !== "ready" && stateRef.current !== "compacting")
      ) {
        return;
      }

      setMessages((previous) => [
        ...previous,
        {
          id: nextLocalId.current--,
          role: "user",
          text,
          from_voice: fromVoice,
          created_at: Date.now(),
        },
      ]);
      streaming.current = "";
      turnUsedTool.current = false;
      setStreamingText("");
      setStreamingTools([]);
      setError(null);
      setTurnError(null);
      applyState("generating");

      sendMessage(
        text,
        (event) => {
          if (!alive.current) return;
          switch (event.type) {
            case "compacting":
              setCompactingReply(true);
              break;
            case "token":
              setCompactingReply(false);
              streaming.current = (streaming.current ?? "") + event.text;
              setStreamingText(streaming.current);
              break;
            case "tool": {
              turnUsedTool.current = true;
              const label = toolLabel(event.name);
              setStreamingTools((previous) =>
                previous.includes(label) ? previous : [...previous, label],
              );
              break;
            }
            case "done":
            case "cancelled":
              finishStream(event.message_id);
              break;
            case "error":
              streaming.current = null;
              turnUsedTool.current = false;
              setStreamingText(null);
              setStreamingTools([]);
              setCompactingReply(false);
              setTurnError(event.message);
              applyState("ready");
              break;
          }
        },
        fromVoice,
      ).catch((failure: unknown) => {
        if (!alive.current) return;
        streaming.current = null;
        turnUsedTool.current = false;
        setStreamingText(null);
        setStreamingTools([]);
        setCompactingReply(false);
        setTurnError(errorMessage(failure));
        applyState("ready");
      });
    },
    [applyState, finishStream],
  );

  const start = useCallback(() => {
    setError(null);
    setTurnError(null);
    applyState("loading");
    startChat()
      .then(() => getChatState())
      .then((current) => {
        if (!alive.current) return;
        applyState(current.state);
        setError(current.error);
      })
      .catch((failure: unknown) => {
        if (!alive.current) return;
        setError(errorMessage(failure));
        applyState("error");
      });
  }, [applyState]);

  const stop = useCallback(() => {
    stopChat()
      .then(() => {
        if (!alive.current) return;
        streaming.current = null;
        turnUsedTool.current = false;
        setStreamingText(null);
        setStreamingTools([]);
        setCompactingReply(false);
        setError(null);
        setTurnError(null);
        applyState("stopped");
      })
      .catch((failure: unknown) => {
        if (!alive.current) return;
        setError(errorMessage(failure));
        applyState("error");
      });
  }, [applyState]);

  const cancel = useCallback(() => {
    cancelGeneration().catch((failure: unknown) => {
      if (!alive.current) return;
      setError(errorMessage(failure));
    });
  }, []);

  return {
    messages,
    chatState,
    error,
    turnError,
    streamingText,
    streamingTools,
    compactingReply,
    start,
    stop,
    send,
    cancel,
  };
}
