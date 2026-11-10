import { useState } from "react";
import {
  Accordion,
  Button,
  Card,
  Chip,
  Input,
  Label,
  Meter,
  Slider,
  TextField,
} from "@heroui/react";
import { requestEstimate } from "../../lib/api.ts";
import type {
  ContextReport,
  Estimate,
  MemoryOverrides,
  ModelRow,
  Settings,
} from "../../lib/api.ts";
import { clamp, formatGb, formatTokens } from "../../lib/format.ts";
import { contextHint, statusColor, usagePercent } from "../../lib/memory.ts";
import type { ChangeSettings } from "./SlotOptions.tsx";

const CONTEXT_MIN = 256;
const CONTEXT_STEP = 256;
const CONTEXT_FALLBACK_MAX = 32768;

export interface MemoryContextCardProps {
  settings: Settings;
  /** The newest forecast, or `null` before the first one arrives. */
  estimate: Estimate | null;
  /** Whether a newer forecast is on its way. */
  pending: boolean;
  onChange: ChangeSettings;
}

/** The Memory & context card: budget, usage, model table and context settings. */
export function MemoryContextCard({
  settings,
  estimate,
  pending,
  onChange,
}: MemoryContextCardProps) {
  if (!estimate) {
    return (
      <Card.Root className="min-h-48 border border-separator shadow-none">
        <Card.Content className="flex flex-col justify-center gap-4 p-4">
          <p className="text-xs text-muted">Estimating…</p>
        </Card.Content>
      </Card.Root>
    );
  }

  const color = statusColor(estimate.status);
  const hasTextModel = Boolean(settings.models.text.path);

  return (
    <Card.Root className="border border-separator shadow-none">
      <Card.Content className="flex flex-col gap-4 p-4">
        <Summary estimate={estimate} pending={pending} color={color} />

        <Meter.Root
          aria-label="RAM use"
          value={usagePercent(estimate.ram_used, estimate.budget.ram)}
          color={color}
          size="sm"
        >
          <Meter.Track>
            <Meter.Fill />
          </Meter.Track>
        </Meter.Root>

        {estimate.context?.limits.fits === false ? (
          <p className="text-sm text-danger">
            This setup doesn't fit in memory. Pick a smaller model or free up
            RAM.
          </p>
        ) : null}

        <ModelTable estimate={estimate} />

        {hasTextModel ? (
          <ContextControls
            settings={settings}
            context={estimate.context}
            onChange={onChange}
          />
        ) : (
          <p className="text-sm text-muted">
            Choose a text model to see how much memory it needs.
          </p>
        )}

        <CompactionControls settings={settings} onChange={onChange} />

        <Advanced
          settings={settings}
          estimate={estimate}
          onChange={onChange}
        />
      </Card.Content>
    </Card.Root>
  );
}

function Summary({
  estimate,
  pending,
  color,
}: {
  estimate: Estimate;
  pending: boolean;
  color: ReturnType<typeof statusColor>;
}) {
  const { budget } = estimate;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-2">
        <Chip size="sm" variant="soft" color={color}>
          {estimate.status_label}
        </Chip>
        <span className="text-sm">
          {formatGb(estimate.ram_used)} of {formatGb(budget.ram)} GB RAM
        </span>
        {pending ? (
          <span className="text-xs text-muted">Estimating…</span>
        ) : null}
      </div>
      {budget.vram > 0 ? (
        <p className="text-xs text-muted">
          {formatGb(estimate.vram_used)} of {formatGb(budget.vram)} GB{" "}
          {budget.unified ? "GPU limit" : "VRAM"}
        </p>
      ) : null}
    </div>
  );
}

function ModelTable({ estimate }: { estimate: Estimate }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-3 text-xs text-muted">
        <span>Model</span>
        <span className="text-right">Weights</span>
        <span className="text-right">Context</span>
        <span className="text-right">Extra</span>
      </div>
      {estimate.rows.map((row) => (
        <ModelTableRow key={row.slot} row={row} />
      ))}
      <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-3 text-sm">
        <span>Regent</span>
        <span />
        <span />
        <span className="text-right tabular-nums">
          {formatGb(estimate.fixed_extras)}
        </span>
      </div>
      <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-3 text-sm font-semibold">
        <span className="col-span-3">Total</span>
        <span className="text-right tabular-nums">
          {formatGb(estimate.ram_used)}
        </span>
      </div>
    </div>
  );
}

function ModelTableRow({ row }: { row: ModelRow }) {
  const label = row.file_name || row.slot;
  if (row.error) {
    return (
      <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-3 text-sm">
        <span className="break-words text-danger">{label}</span>
        <span className="text-right tabular-nums">—</span>
        <span className="text-right tabular-nums">—</span>
        <span className="text-right tabular-nums">—</span>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-3 text-sm">
      <span className="break-words">{label}</span>
      <span className="text-right tabular-nums">{formatGb(row.weights)}</span>
      <span className="text-right tabular-nums">
        {row.kv_unknown ? "?" : formatTokens(row.context)}
      </span>
      <span className="text-right tabular-nums">{formatGb(row.extra)}</span>
    </div>
  );
}

function ContextControls({
  settings,
  context,
  onChange,
}: {
  settings: Settings;
  context: ContextReport | null;
  onChange: ChangeSettings;
}) {
  const saved = settings.models.text.context_size;
  // The draft remembers the saved value it was based on, so a save (or a
  // reset) makes it stale and the saved value takes over again.
  const [draft, setDraft] = useState<{
    base: number;
    value: number;
  } | null>(null);
  const [previewBase, setPreviewBase] = useState<
    ContextReport | null | undefined
  >(undefined);

  const limits = context?.limits ?? null;
  const rawMax =
    limits?.effective_max ??
    limits?.model_max ??
    CONTEXT_FALLBACK_MAX;
  const max = Math.max(CONTEXT_MIN, rawMax);
  const value = clamp(
    draft?.base === saved ? draft.value : saved,
    CONTEXT_MIN,
    max,
  );
  const hint = contextHint(context);
  const previewing = previewBase !== undefined && previewBase === context;

  function preview(next: number) {
    setPreviewBase(context);
    void requestEstimate({
      ...settings,
      models: {
        ...settings.models,
        text: { ...settings.models.text, context_size: next },
      },
    }).catch(() => {
      setPreviewBase(undefined);
    });
  }

  function change(next: number | number[]) {
    const size = Array.isArray(next) ? next[0] : next;
    setDraft({ base: saved, value: size });
    preview(size);
  }

  function commit(next: number | number[]) {
    const size = Array.isArray(next) ? next[0] : next;
    setDraft({ base: saved, value: size });
    onChange(
      {
        ...settings,
        models: {
          ...settings.models,
          text: { ...settings.models.text, context_size: size },
        },
      },
      { debounce: true },
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium">Context size</span>
        <span className="text-xs text-muted tabular-nums">
          {formatTokens(value)} tokens
          {previewing ? " · Estimating…" : null}
        </span>
      </div>
      <Slider
        aria-label="Context size"
        minValue={CONTEXT_MIN}
        maxValue={max}
        step={CONTEXT_STEP}
        value={value}
        onChange={change}
        onChangeEnd={commit}
      >
        <Slider.Track>
          <Slider.Fill />
          <Slider.Thumb />
        </Slider.Track>
      </Slider>
      <p className="text-xs text-muted">
        Model max{" "}
        {limits?.model_max == null ? "unknown" : formatTokens(limits.model_max)}
        {" · "}
        This PC max{" "}
        {limits?.system_max == null
          ? "unknown"
          : formatTokens(limits.system_max)}
      </p>
      {hint ? <p className="text-xs text-warning">{hint}</p> : null}
    </div>
  );
}

function CompactionControls({
  settings,
  onChange,
}: {
  settings: Settings;
  onChange: ChangeSettings;
}) {
  const savedTrigger = clamp(settings.compaction.trigger_percent, 50, 95);
  const savedTarget = clamp(settings.compaction.target_percent, 20, 80);
  // As with the context slider, the draft is only used while the saved values
  // it was built from are still the current ones.
  const [draft, setDraft] = useState<{
    baseTrigger: number;
    baseTarget: number;
    trigger: number;
    target: number;
  } | null>(null);

  const active =
    draft &&
    draft.baseTrigger === savedTrigger &&
    draft.baseTarget === savedTarget
      ? draft
      : null;
  const trigger = active?.trigger ?? savedTrigger;
  const target = active?.target ?? savedTarget;
  const invalid = target >= trigger;

  function changeTrigger(next: number | number[]) {
    const value = Array.isArray(next) ? next[0] : next;
    setDraft({
      baseTrigger: savedTrigger,
      baseTarget: savedTarget,
      trigger: value,
      target,
    });
    if (target < value) {
      onChange(
        {
          ...settings,
          compaction: { trigger_percent: value, target_percent: target },
        },
        { debounce: true },
      );
    }
  }

  function changeTarget(next: number | number[]) {
    const value = Array.isArray(next) ? next[0] : next;
    setDraft({
      baseTrigger: savedTrigger,
      baseTarget: savedTarget,
      trigger,
      target: value,
    });
    if (value < trigger) {
      onChange(
        {
          ...settings,
          compaction: { trigger_percent: trigger, target_percent: value },
        },
        { debounce: true },
      );
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium">Compact at</span>
          <span className="text-xs text-muted tabular-nums">{trigger}%</span>
        </div>
        <Slider
          aria-label="Compact at"
          minValue={50}
          maxValue={95}
          step={5}
          value={trigger}
          onChange={changeTrigger}
        >
          <Slider.Track>
            <Slider.Fill />
            <Slider.Thumb />
          </Slider.Track>
        </Slider>
      </div>

      <div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-medium">Compact down to</span>
          <span className="text-xs text-muted tabular-nums">{target}%</span>
        </div>
        <Slider
          aria-label="Compact down to"
          minValue={20}
          maxValue={80}
          step={5}
          value={target}
          onChange={changeTarget}
        >
          <Slider.Track>
            <Slider.Fill />
            <Slider.Thumb />
          </Slider.Track>
        </Slider>
      </div>

      {invalid ? (
        <p className="text-xs text-danger">
          Must be lower than "Compact at".
        </p>
      ) : (
        <p className="text-xs text-muted">
          When the chat reaches {trigger}% of the context, older messages are
          summarised down to {target}%.
        </p>
      )}
    </div>
  );
}

function Advanced({
  settings,
  estimate,
  onChange,
}: {
  settings: Settings;
  estimate: Estimate;
  onChange: ChangeSettings;
}) {
  const { cpu, gpus, ram } = estimate.hardware;
  const [resetToken, setResetToken] = useState(0);
  const detectedVram = gpus.reduce(
    (sum, gpu) => sum + (gpu.dedicated_vram ?? 0),
    0,
  );

  const parts: string[] = [];
  if (cpu.name) parts.push(cpu.name);
  parts.push(`${cpu.logical_threads} threads`);
  for (const gpu of gpus) {
    parts.push(
      gpu.dedicated_vram == null
        ? gpu.name
        : `${gpu.name} (${formatGb(gpu.dedicated_vram)} GB)`,
    );
  }

  function setOverride(key: keyof MemoryOverrides, value: number | null) {
    onChange(
      { ...settings, memory: { ...settings.memory, [key]: value } },
      { debounce: true },
    );
  }

  function clearOverrides() {
    setResetToken((token) => token + 1);
    onChange(
      { ...settings, memory: { ram_gb: null, vram_gb: null } },
      { debounce: false },
    );
  }

  return (
    <Accordion hideSeparator>
      <Accordion.Item id="advanced">
        <Accordion.Heading>
          <Accordion.Trigger className="px-0 py-1 text-xs text-muted">
            Advanced
            <Accordion.Indicator />
          </Accordion.Trigger>
        </Accordion.Heading>
        <Accordion.Panel>
          <div className="flex flex-col gap-3 pt-1 pb-2">
            <div className="flex gap-2">
              <GbInput
                key={`ram-${resetToken}`}
                label="RAM override"
                value={settings.memory.ram_gb ?? null}
                detected={formatGb(ram.total)}
                onChange={(value) => setOverride("ram_gb", value)}
              />
              <GbInput
                key={`vram-${resetToken}`}
                label="VRAM override"
                value={settings.memory.vram_gb ?? null}
                detected={formatGb(detectedVram)}
                onChange={(value) => setOverride("vram_gb", value)}
              />
            </div>
            <div>
              <Button size="sm" variant="ghost" onPress={clearOverrides}>
                Use detected
              </Button>
            </div>
            <p className="text-xs text-muted">{parts.join(" · ")}</p>
          </div>
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

function GbInput({
  label,
  value,
  detected,
  onChange,
}: {
  label: string;
  value: number | null;
  detected: string;
  onChange: (value: number | null) => void;
}) {
  // `null` means "show the saved value"; otherwise the user's in-progress text.
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (value == null ? "" : String(value));

  function change(text: string) {
    setDraft(text);
    const trimmed = text.trim();
    if (!trimmed) {
      onChange(null);
      return;
    }
    const parsed = Number.parseFloat(trimmed);
    if (Number.isFinite(parsed) && parsed > 0) {
      onChange(Math.round(parsed * 10) / 10);
    }
  }

  return (
    <TextField className="w-full">
      <Label className="text-xs font-medium">{label}</Label>
      <Input
        type="number"
        inputMode="decimal"
        min={0.1}
        step={0.1}
        placeholder={`Detected: ${detected}`}
        value={shown}
        onChange={(event) => change(event.target.value)}
        onBlur={() => setDraft(null)}
        className="h-8 w-full text-sm"
      />
    </TextField>
  );
}
