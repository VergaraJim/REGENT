import { useEffect, useState } from "react";
import {
  Button,
  Label,
  Separator,
  Spinner,
  Switch,
  Tabs,
  TextArea,
  TextField,
} from "@heroui/react";
import { ChevronLeft } from "lucide-react";
import { Section } from "../components/settings/Section.tsx";
import { StatusBanner } from "../components/settings/StatusBanner.tsx";
import { ModelSlotRow } from "../components/settings/ModelSlotRow.tsx";
import { MemoryContextCard } from "../components/settings/MemoryContextCard.tsx";
import { SlotOptions, optionSlotId } from "../components/settings/SlotOptions.tsx";
import { useEstimate } from "../hooks/useEstimate.ts";
import type { SettingsController } from "../hooks/useSettings.ts";
import type { ModelRow, Settings, SlotId, Theme } from "../lib/api.ts";

const MAX_INSTRUCTIONS = 2000;

type OptionsSlot = "text" | "vision" | "embedding" | "stt";

export function SettingsView({
  onBack,
  controller,
}: {
  onBack: () => void;
  controller: SettingsController;
}) {
  const { view, saveError, update, reset } = controller;
  const settings = view?.settings ?? null;
  const { estimate, pending } = useEstimate(settings);
  const [spinnerVisible, setSpinnerVisible] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => setSpinnerVisible(true), 200);
    return () => window.clearTimeout(timer);
  }, []);

  const rows = new Map<SlotId, ModelRow>(
    (estimate?.rows ?? []).map((row) => [row.slot, row]),
  );

  function setPath(slot: SlotId, path: string | null) {
    if (!settings) return;
    update(withPath(settings, slot, path));
  }

  function removePath(slot: SlotId) {
    setPath(slot, null);
  }

  function rowError(slot: SlotId): string | null {
    return rows.get(slot)?.error ?? null;
  }

  function optionsFor(slot: OptionsSlot) {
    if (!settings) return null;
    return (
      <SlotOptions
        slot={slot}
        settings={settings}
        layerCount={rows.get(optionSlotId(slot))?.layer_count ?? null}
        onChange={update}
      />
    );
  }

  return (
    <>
      <header className="flex h-12 shrink-0 items-center border-b border-separator px-4">
        <span className="flex items-center gap-2">
          <Button
            isIconOnly
            variant="ghost"
            size="sm"
            aria-label="Back"
            onPress={onBack}
          >
            <ChevronLeft size={18} aria-hidden="true" />
          </Button>
          <span className="text-base font-semibold">Settings</span>
        </span>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto p-4">
        {view && settings ? (
          <div className="flex flex-col gap-6">
            <StatusBanner
              loadError={view.load_error}
              savingBlocked={view.saving_blocked}
              saveError={saveError}
              onReset={reset}
            />

            <Section title="Models">
              <ModelSlotRow
                title="Text model"
                files={[
                  {
                    slot: "text",
                    path: settings.models.text.path,
                    error: rowError("text"),
                  },
                ]}
                onPick={setPath}
                onRemove={removePath}
                options={
                  settings.models.text.path ? optionsFor("text") : undefined
                }
              />
              <Separator />
              <ModelSlotRow
                title="Vision model"
                files={[
                  {
                    label: "Model",
                    slot: "vision",
                    path: settings.models.vision.model_path,
                    error: rowError("vision"),
                  },
                  {
                    label: "Projector",
                    slot: "vision_projector",
                    path: settings.models.vision.projector_path,
                    error: rowError("vision_projector"),
                  },
                ]}
                onPick={setPath}
                onRemove={removePath}
                options={
                  settings.models.vision.model_path
                    ? optionsFor("vision")
                    : undefined
                }
              />
              <Separator />
              <ModelSlotRow
                title="Embedding model"
                files={[
                  {
                    slot: "embedding",
                    path: settings.models.embedding.path,
                    error: rowError("embedding"),
                  },
                ]}
                onPick={setPath}
                onRemove={removePath}
                options={
                  settings.models.embedding.path
                    ? optionsFor("embedding")
                    : undefined
                }
              />
              <Separator />
              <ModelSlotRow
                title="Speech-to-text"
                files={[
                  {
                    slot: "speech_to_text",
                    path: settings.models.stt.path,
                    error: rowError("speech_to_text"),
                  },
                ]}
                onPick={setPath}
                onRemove={removePath}
                options={
                  settings.models.stt.path ? optionsFor("stt") : undefined
                }
              />
              <Separator />
              <ModelSlotRow
                title="Voice"
                files={[
                  {
                    slot: "text_to_speech",
                    path: settings.models.tts.voice_path,
                    error: rowError("text_to_speech"),
                  },
                ]}
                onPick={setPath}
                onRemove={removePath}
              />
            </Section>

            <Section title="Memory & context">
              <MemoryContextCard
                settings={settings}
                estimate={estimate}
                pending={pending}
                onChange={update}
              />
            </Section>

            <Section title="Voice">
              <div className="py-3">
                <Switch
                  isSelected={settings.voice.speak_replies}
                  onChange={(speak_replies) =>
                    update({
                      ...settings,
                      voice: { ...settings.voice, speak_replies },
                    })
                  }
                >
                  <Switch.Content className="flex w-full items-center justify-between gap-3">
                    <Label>Read replies aloud</Label>
                    <Switch.Control>
                      <Switch.Thumb />
                    </Switch.Control>
                  </Switch.Content>
                </Switch>
              </div>
              <Separator />
              <div className="py-3">
                <Switch
                  isSelected={settings.voice.pause_mic_while_speaking}
                  onChange={(pause_mic_while_speaking) =>
                    update({
                      ...settings,
                      voice: { ...settings.voice, pause_mic_while_speaking },
                    })
                  }
                >
                  <Switch.Content className="flex w-full items-center justify-between gap-3">
                    <Label>Pause microphone while speaking</Label>
                    <Switch.Control>
                      <Switch.Thumb />
                    </Switch.Control>
                  </Switch.Content>
                </Switch>
              </div>
            </Section>

            <Section title="Assistant">
              <div className="py-3">
                <TextField className="w-full" fullWidth>
                  <Label className="text-xs font-medium">
                    Extra instructions
                  </Label>
                  <TextArea
                    rows={4}
                    maxLength={MAX_INSTRUCTIONS}
                    placeholder="e.g. Keep answers short."
                    value={settings.system_prompt_addition}
                    onChange={(event) =>
                      update(
                        {
                          ...settings,
                          system_prompt_addition: event.target.value.slice(
                            0,
                            MAX_INSTRUCTIONS,
                          ),
                        },
                        { debounce: true },
                      )
                    }
                    className="w-full text-sm"
                  />
                  <p className="self-end text-xs text-muted tabular-nums">
                    {settings.system_prompt_addition.length}/{MAX_INSTRUCTIONS}
                  </p>
                </TextField>
              </div>
            </Section>

            <Section title="Appearance">
              <div className="flex flex-col gap-1 py-3">
                <span className="text-xs font-medium text-muted">Theme</span>
                <Tabs
                  selectedKey={settings.appearance.theme}
                  onSelectionChange={(key) =>
                    update({
                      ...settings,
                      appearance: {
                        ...settings.appearance,
                        theme: String(key) as Theme,
                      },
                    })
                  }
                >
                  <Tabs.List aria-label="Theme">
                    <Tabs.Tab id="system">System</Tabs.Tab>
                    <Tabs.Tab id="light">Light</Tabs.Tab>
                    <Tabs.Tab id="dark">Dark</Tabs.Tab>
                  </Tabs.List>
                </Tabs>
              </div>
              <Separator />
              <div className="py-3">
                <Switch
                  isSelected={settings.appearance.always_on_top}
                  onChange={(always_on_top) =>
                    update({
                      ...settings,
                      appearance: { ...settings.appearance, always_on_top },
                    })
                  }
                >
                  <Switch.Content className="flex w-full items-center justify-between gap-3">
                    <Label>Keep window on top</Label>
                    <Switch.Control>
                      <Switch.Thumb />
                    </Switch.Control>
                  </Switch.Content>
                </Switch>
              </div>
            </Section>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <StatusBanner
              loadError={null}
              savingBlocked={false}
              saveError={saveError}
              onReset={reset}
            />
            {spinnerVisible ? (
              <div className="flex justify-center py-8">
                <Spinner size="sm" />
              </div>
            ) : null}
          </div>
        )}
      </main>
    </>
  );
}

/** Returns a copy of `settings` with `slot` pointing at `path` (or cleared). */
function withPath(
  settings: Settings,
  slot: SlotId,
  path: string | null,
): Settings {
  const { models } = settings;
  switch (slot) {
    case "text":
      return {
        ...settings,
        models: { ...models, text: { ...models.text, path } },
      };
    case "vision":
      return {
        ...settings,
        models: { ...models, vision: { ...models.vision, model_path: path } },
      };
    case "vision_projector":
      return {
        ...settings,
        models: {
          ...models,
          vision: { ...models.vision, projector_path: path },
        },
      };
    case "embedding":
      return {
        ...settings,
        models: { ...models, embedding: { ...models.embedding, path } },
      };
    case "speech_to_text":
      return {
        ...settings,
        models: { ...models, stt: { ...models.stt, path } },
      };
    case "text_to_speech":
      return {
        ...settings,
        models: { ...models, tts: { voice_path: path } },
      };
  }
}
