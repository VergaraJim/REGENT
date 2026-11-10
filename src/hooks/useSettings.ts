import { useCallback, useEffect, useRef, useState } from "react";
import {
  getSettings,
  resetSettings,
  updateSettings,
} from "../lib/api.ts";
import type { Settings, SettingsView } from "../lib/api.ts";
import { errorMessage } from "../lib/format.ts";

/** How long text and number inputs wait before their save is sent. */
export const SAVE_DELAY_MS = 400;

export interface UpdateOptions {
  /** Wait for `SAVE_DELAY_MS` since the last call before saving. */
  debounce?: boolean;
}

export interface SettingsController {
  /** The settings and the state of their file; `null` until the first load. */
  view: SettingsView | null;
  /** The last save or load failure, if any. */
  saveError: string | null;
  /** Applies `next` optimistically and saves it. */
  update: (next: Settings, options?: UpdateOptions) => void;
  /** Restores the defaults, backing up an unreadable file first. */
  reset: () => void;
}

export function useSettings(): SettingsController {
  const [view, setView] = useState<SettingsView | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const viewRef = useRef<SettingsView | null>(null);
  const timer = useRef<number | null>(null);
  const request = useRef(0);
  const alive = useRef(true);

  const applyView = useCallback((next: SettingsView) => {
    viewRef.current = next;
    setView(next);
  }, []);

  const clearTimer = useCallback(() => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  const save = useCallback(
    (settings: Settings, requestId: number) => {
      updateSettings(settings)
        .then((saved) => {
          if (!alive.current || requestId !== request.current) return;
          applyView(saved);
          setSaveError(null);
        })
        .catch((error: unknown) => {
          if (!alive.current || requestId !== request.current) return;
          setSaveError(errorMessage(error));
        });
    },
    [applyView],
  );

  useEffect(() => {
    alive.current = true;
    let cancelled = false;
    getSettings()
      .then((loaded) => {
        if (cancelled) return;
        applyView(loaded);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setSaveError(errorMessage(error));
      });
    return () => {
      cancelled = true;
    };
  }, [applyView]);

  useEffect(() => {
    return () => {
      alive.current = false;
      clearTimer();
    };
  }, [clearTimer]);

  const update = useCallback(
    (next: Settings, options?: UpdateOptions) => {
      const current = viewRef.current;
      if (!current) return;
      const requestId = ++request.current;
      applyView({ ...current, settings: next });
      clearTimer();
      if (options?.debounce) {
        timer.current = window.setTimeout(() => {
          timer.current = null;
          save(next, requestId);
        }, SAVE_DELAY_MS);
      } else {
        save(next, requestId);
      }
    },
    [applyView, clearTimer, save],
  );

  const reset = useCallback(() => {
    const requestId = ++request.current;
    clearTimer();
    resetSettings()
      .then((restored) => {
        if (!alive.current || requestId !== request.current) return;
        applyView(restored);
        setSaveError(null);
      })
      .catch((error: unknown) => {
        if (!alive.current || requestId !== request.current) return;
        setSaveError(errorMessage(error));
      });
  }, [applyView, clearTimer]);

  return { view, saveError, update, reset };
}
