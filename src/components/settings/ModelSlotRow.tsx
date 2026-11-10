import { useState } from "react";
import { Button, Tooltip } from "@heroui/react";
import { X } from "lucide-react";
import { pickModelFile } from "../../lib/api.ts";
import type { SlotId } from "../../lib/api.ts";
import { errorMessage, fileNameFromPath } from "../../lib/format.ts";
import type { ReactNode } from "react";

export interface ModelFile {
  /** The small caption above the file name, used when a row has several files. */
  label?: string;
  slot: SlotId;
  path?: string | null;
  /** The estimate's complaint about this file, if it has one. */
  error?: string | null;
}

export interface ModelFileLineProps {
  label: string;
  slot: SlotId;
  path?: string | null;
  error?: string | null;
  /** Large label for a single-file row, small caption for a sub-line. */
  title?: boolean;
  onPick: (slot: SlotId, path: string) => void;
  onRemove: (slot: SlotId) => void;
}

/** One file: its label, name, path tooltip and the choose/remove buttons. */
export function ModelFileLine({
  label,
  slot,
  path,
  error,
  title = false,
  onPick,
  onRemove,
}: ModelFileLineProps) {
  const [warning, setWarning] = useState<string | null>(null);
  const [picking, setPicking] = useState(false);
  const name = path ? fileNameFromPath(path) : "";

  async function choose() {
    setPicking(true);
    try {
      const picked = await pickModelFile(slot);
      if (!picked) return;
      setWarning(picked.warning);
      onPick(slot, picked.path);
    } catch (error: unknown) {
      setWarning(errorMessage(error));
    } finally {
      setPicking(false);
    }
  }

  return (
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0 flex-1">
        <div
          className={
            title ? "text-sm font-medium" : "text-xs font-medium text-muted"
          }
        >
          {label}
        </div>
        {name ? (
          <Tooltip delay={400}>
            <Tooltip.Trigger className="mt-0.5 block w-full" tabIndex={0}>
              <span className="block truncate text-xs text-muted">{name}</span>
            </Tooltip.Trigger>
            <Tooltip.Content>{path}</Tooltip.Content>
          </Tooltip>
        ) : (
          <p className="mt-0.5 text-xs text-muted">Not set</p>
        )}
        {error ? (
          <p className="mt-1 text-xs text-danger">{error}</p>
        ) : null}
        {warning ? (
          <p className="mt-1 text-xs text-warning">{warning}</p>
        ) : null}
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <Button
          size="sm"
          variant="secondary"
          isDisabled={picking}
          aria-label={`Choose ${label}`}
          onPress={choose}
        >
          Choose…
        </Button>
        {name ? (
          <Button
            isIconOnly
            size="sm"
            variant="ghost"
            aria-label="Remove"
            onPress={() => {
              setWarning(null);
              onRemove(slot);
            }}
          >
            <X size={16} aria-hidden="true" />
          </Button>
        ) : null}
      </div>
    </div>
  );
}

export interface ModelSlotRowProps {
  title: string;
  files: ModelFile[];
  onPick: (slot: SlotId, path: string) => void;
  onRemove: (slot: SlotId) => void;
  /** Per-model options, rendered under the file names when a file is set. */
  options?: ReactNode;
}

/** A settings row for one model slot, or for a row with several files. */
export function ModelSlotRow({
  title,
  files,
  onPick,
  onRemove,
  options,
}: ModelSlotRowProps) {
  const single = files.length === 1;

  return (
    <div className="py-3">
      {single ? null : (
        <h3 className="text-sm font-medium">{title}</h3>
      )}
      <div className="flex flex-col gap-2">
        {files.map((file) => (
          <ModelFileLine
            key={file.slot}
            label={single ? title : (file.label ?? "Model")}
            slot={file.slot}
            path={file.path}
            error={file.error}
            title={single}
            onPick={onPick}
            onRemove={onRemove}
          />
        ))}
      </div>
      {options ? <div className="mt-1">{options}</div> : null}
    </div>
  );
}
