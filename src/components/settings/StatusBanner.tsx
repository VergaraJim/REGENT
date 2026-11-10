import { useState } from "react";
import { Alert, Button } from "@heroui/react";

export interface StatusBannerProps {
  loadError: string | null;
  savingBlocked: boolean;
  saveError: string | null;
  onReset: () => void;
}

/** The warnings at the top of the settings body. */
export function StatusBanner({
  loadError,
  savingBlocked,
  saveError,
  onReset,
}: StatusBannerProps) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="flex flex-col gap-2">
      {loadError ? (
        <Alert status="warning" className="text-sm">
          <Alert.Indicator />
          <Alert.Content>
            <Alert.Description>
              Your settings file couldn't be read, so Regent is using
              defaults. Your file hasn't been changed.
              {savingBlocked
                ? " Changes aren't saved until you reset."
                : null}
            </Alert.Description>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                variant="ghost"
                onPress={() => setShowDetails((shown) => !shown)}
              >
                {showDetails ? "Hide details" : "Show details"}
              </Button>
              <Button size="sm" variant="primary" onPress={onReset}>
                Reset to defaults
              </Button>
            </div>
            {showDetails ? (
              <p className="mt-2 text-xs break-words text-muted">{loadError}</p>
            ) : null}
          </Alert.Content>
        </Alert>
      ) : null}

      {saveError ? (
        <Alert status="danger" className="text-sm">
          <Alert.Indicator />
          <Alert.Content>
            <Alert.Description>{saveError}</Alert.Description>
          </Alert.Content>
        </Alert>
      ) : null}
    </div>
  );
}
