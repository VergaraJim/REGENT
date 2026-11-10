import { useState } from "react";
import {
  Accordion,
  Input,
  Label,
  ListBox,
  ListBoxItem,
  Select,
  Slider,
  TextField,
} from "@heroui/react";
import type { Settings, SlotId } from "../../lib/api.ts";
import { clamp } from "../../lib/format.ts";

/** Saves `next`, optionally after the text/number input debounce. */
export type ChangeSettings = (
  next: Settings,
  options?: { debounce?: boolean },
) => void;

type OptionSlot = "text" | "vision" | "embedding" | "stt";

interface SlotOptionsProps {
  slot: OptionSlot;
  settings: Settings;
  /** The newest estimate's layer count for this slot, or `null` when unknown. */
  layerCount: number | null;
  onChange: ChangeSettings;
}

/** The collapsible per-model options for one model row. */
export function SlotOptions({
  slot,
  settings,
  layerCount,
  onChange,
}: SlotOptionsProps) {
  const { models } = settings;

  const setText = (patch: Partial<Settings["models"]["text"]>) => {
    onChange(
      {
        ...settings,
        models: { ...models, text: { ...models.text, ...patch } },
      },
      { debounce: true },
    );
  };

  const setVision = (patch: Partial<Settings["models"]["vision"]>) => {
    onChange(
      {
        ...settings,
        models: { ...models, vision: { ...models.vision, ...patch } },
      },
      { debounce: true },
    );
  };

  const setEmbedding = (patch: Partial<Settings["models"]["embedding"]>) => {
    onChange(
      {
        ...settings,
        models: { ...models, embedding: { ...models.embedding, ...patch } },
      },
      { debounce: true },
    );
  };

  const setLanguage = (language: string) => {
    onChange(
      { ...settings, models: { ...models, stt: { ...models.stt, language } } },
      { debounce: false },
    );
  };

  return (
    <Accordion hideSeparator>
      <Accordion.Item id="options">
        <Accordion.Heading>
          <Accordion.Trigger className="px-0 py-1 text-xs text-muted">
            Options
            <Accordion.Indicator />
          </Accordion.Trigger>
        </Accordion.Heading>
        <Accordion.Panel>
          <div className="flex flex-col gap-3 pt-1 pb-2">
            {slot === "text" ? (
              <GpuLayers
                value={models.text.gpu_layers}
                layerCount={layerCount}
                onChange={(gpu_layers) => setText({ gpu_layers })}
              />
            ) : null}

            {slot === "vision" ? (
              <>
                <GpuLayers
                  value={models.vision.gpu_layers}
                  layerCount={layerCount}
                  onChange={(gpu_layers) => setVision({ gpu_layers })}
                />
                <NumberOption
                  label="Max image size"
                  value={models.vision.max_image_edge}
                  min={256}
                  max={4096}
                  step={64}
                  suffix="px"
                  onChange={(max_image_edge) => setVision({ max_image_edge })}
                />
                <NumberOption
                  label="Unload after"
                  value={models.vision.keep_alive_secs}
                  min={0}
                  max={3600}
                  step={1}
                  suffix="seconds"
                  onChange={(keep_alive_secs) => setVision({ keep_alive_secs })}
                />
              </>
            ) : null}

            {slot === "embedding" ? (
              <>
                <GpuLayers
                  value={models.embedding.gpu_layers}
                  layerCount={layerCount}
                  onChange={(gpu_layers) => setEmbedding({ gpu_layers })}
                />
                <NumberOption
                  label="Context"
                  value={models.embedding.context_size}
                  min={256}
                  max={32768}
                  step={256}
                  onChange={(context_size) => setEmbedding({ context_size })}
                />
                <TextOption
                  label="Document prefix"
                  value={models.embedding.document_prefix ?? ""}
                  placeholder="Automatic"
                  onChange={(document_prefix) =>
                    setEmbedding({ document_prefix: document_prefix || null })
                  }
                />
                <TextOption
                  label="Query prefix"
                  value={models.embedding.query_prefix ?? ""}
                  placeholder="Automatic"
                  onChange={(query_prefix) =>
                    setEmbedding({ query_prefix: query_prefix || null })
                  }
                />
              </>
            ) : null}

            {slot === "stt" ? (
              <LanguageSelect value={models.stt.language} onChange={setLanguage} />
            ) : null}
          </div>
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

function GpuLayers({
  value,
  layerCount,
  onChange,
}: {
  value: number;
  layerCount: number | null;
  onChange: (value: number) => void;
}) {
  const known = layerCount !== null && layerCount > 0;
  const max = known ? layerCount : 1;
  const current = clamp(value, 0, max);

  return (
    <div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium">GPU layers</span>
        <span className="text-xs text-muted tabular-nums">
          {known ? current : "—"}
        </span>
      </div>
      <Slider
        aria-label="GPU layers"
        minValue={0}
        maxValue={max}
        step={1}
        value={current}
        isDisabled={!known}
        onChange={(next) => onChange(Array.isArray(next) ? next[0] : next)}
        className="mt-1"
      >
        <Slider.Track>
          <Slider.Fill />
          <Slider.Thumb />
        </Slider.Track>
        <Slider.Marks>
          <Slider.Mark value={0}>None</Slider.Mark>
          <Slider.Mark value={max}>All</Slider.Mark>
        </Slider.Marks>
      </Slider>
      {known ? null : (
        <p className="mt-1 text-xs text-muted">Pick a model to set this.</p>
      )}
    </div>
  );
}

function NumberOption({
  label,
  value,
  min,
  max,
  step,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  onChange: (value: number) => void;
}) {
  // `null` means "show the saved value"; otherwise the user's in-progress text.
  const [draft, setDraft] = useState<string | null>(null);

  function change(text: string) {
    setDraft(text);
    const parsed = Number.parseInt(text, 10);
    if (Number.isFinite(parsed) && parsed >= min && parsed <= max) {
      onChange(parsed);
    }
  }

  function commit() {
    const parsed = Number.parseInt(draft ?? String(value), 10);
    const next = Number.isFinite(parsed) ? clamp(parsed, min, max) : value;
    setDraft(null);
    if (next !== value) onChange(next);
  }

  return (
    <TextField className="w-full">
      <Label className="text-xs font-medium">
        {label}
        {suffix ? <span className="font-normal text-muted"> ({suffix})</span> : null}
      </Label>
      <Input
        type="number"
        inputMode="numeric"
        min={min}
        max={max}
        step={step}
        value={draft ?? String(value)}
        onChange={(event) => change(event.target.value)}
        onBlur={commit}
        className="h-8 w-full text-sm"
      />
    </TextField>
  );
}

function TextOption({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
}) {
  return (
    <TextField className="w-full">
      <Label className="text-xs font-medium">{label}</Label>
      <Input
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        className="h-8 w-full text-sm"
      />
    </TextField>
  );
}

const LANGUAGES = [
  { id: "auto", label: "Auto-detect" },
  { id: "en", label: "English" },
  { id: "es", label: "Spanish" },
  { id: "fr", label: "French" },
  { id: "de", label: "German" },
  { id: "it", label: "Italian" },
  { id: "pt", label: "Portuguese" },
  { id: "nl", label: "Dutch" },
  { id: "ru", label: "Russian" },
  { id: "zh", label: "Chinese" },
  { id: "ja", label: "Japanese" },
  { id: "ko", label: "Korean" },
] as const;

function LanguageSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  const selected = LANGUAGES.some((language) => language.id === value)
    ? value
    : "auto";

  return (
    <div className="flex flex-col gap-1">
      <Select
        selectedKey={selected}
        onSelectionChange={(key) => onChange(String(key))}
        className="w-full"
      >
        <Label className="text-xs font-medium">Language</Label>
        <Select.Trigger className="h-8 w-full py-1 text-sm">
          <Select.Value />
          <Select.Indicator />
        </Select.Trigger>
        <Select.Popover>
          <ListBox items={LANGUAGES}>
            {(item) => (
              <ListBoxItem id={item.id}>{item.label}</ListBoxItem>
            )}
          </ListBox>
        </Select.Popover>
      </Select>
      <p className="text-xs text-muted">English-only models need English.</p>
    </div>
  );
}

/** The option slot name used by the estimate rows for a model row. */
export function optionSlotId(slot: OptionSlot): SlotId {
  return slot === "stt" ? "speech_to_text" : slot;
}
