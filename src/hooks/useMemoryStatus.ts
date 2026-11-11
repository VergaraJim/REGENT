import { useEffect, useState } from "react";
import { getMemoryStatus, onMemoryStatus } from "../lib/api.ts";
import type { MemoryStatus } from "../lib/api.ts";

/**
 * Keeps the memory index's state in sync with the backend.
 *
 * Create it once (in `App`) so the footer does not re-subscribe on every
 * render. A failed initial read leaves the status `null`, which renders no line.
 */
export function useMemoryStatus(): MemoryStatus | null {
  const [status, setStatus] = useState<MemoryStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | undefined;

    getMemoryStatus()
      .then((next) => {
        if (!cancelled) setStatus(next);
      })
      .catch(() => {
        // No memory model is a normal state, not an error to show.
      });

    void onMemoryStatus((next) => {
      if (!cancelled) setStatus(next);
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

  return status;
}
