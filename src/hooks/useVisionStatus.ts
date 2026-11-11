import { useEffect, useState } from "react";
import { getVisionStatus, onVisionStatus } from "../lib/api.ts";
import type { VisionStatus } from "../lib/api.ts";

/**
 * Keeps the vision engine's state in sync with the backend.
 *
 * Create it once (in `App`) so the footer does not re-subscribe on every
 * render. A failed initial read leaves the status `null`, which renders no line.
 */
export function useVisionStatus(): VisionStatus | null {
  const [status, setStatus] = useState<VisionStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | undefined;

    getVisionStatus()
      .then((next) => {
        if (!cancelled) setStatus(next);
      })
      .catch(() => {
        // No vision model is a normal state, not an error to show.
      });

    void onVisionStatus((next) => {
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
