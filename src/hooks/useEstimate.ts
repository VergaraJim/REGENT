import { useEffect, useMemo, useRef, useState } from "react";
import { createLatestOnly, onEstimate, requestEstimate } from "../lib/api.ts";
import type { Estimate, EstimateReady, Settings } from "../lib/api.ts";
import { estimateKey } from "../lib/format.ts";

export interface EstimateState {
  /** The newest forecast, or `null` before the first one arrives. */
  estimate: Estimate | null;
  /** Whether a requested forecast is still on its way. */
  pending: boolean;
}

/**
 * Subscribes to forecast events once and asks for a new forecast whenever the
 * settings that change memory use change. Results older than the newest
 * request are ignored.
 */
export function useEstimate(settings: Settings | null): EstimateState {
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [settledKey, setSettledKey] = useState<string | null>(null);
  const latest = useRef(createLatestOnly());
  const key = useMemo(() => estimateKey(settings), [settings]);
  const currentKey = useRef(key);
  const lastRequested = useRef<string | null>(null);

  useEffect(() => {
    currentKey.current = key;
  }, [key]);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | undefined;
    void onEstimate((ready: EstimateReady) => {
      if (!latest.current.accept(ready.generation)) return;
      setEstimate(ready.estimate);
      setSettledKey(currentKey.current);
    }).then((stop) => {
      if (cancelled) {
        stop();
      } else {
        unlisten = stop;
      }
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (!settings || lastRequested.current === key) return;
    lastRequested.current = key;
    let cancelled = false;
    requestEstimate(settings)
      .then((generation) => {
        if (cancelled) return;
        latest.current.track(generation);
      })
      .catch(() => {
        if (!cancelled) setSettledKey(key);
      });
    return () => {
      cancelled = true;
    };
  }, [key, settings]);

  return { estimate, pending: settings !== null && settledKey !== key };
}
