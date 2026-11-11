import { useState } from "react";
import { AppHeader } from "./components/AppHeader.tsx";
import { ChatView } from "./views/ChatView.tsx";
import { SettingsView } from "./views/SettingsView.tsx";
import { useChat } from "./hooks/useChat.ts";
import { useMemoryStatus } from "./hooks/useMemoryStatus.ts";
import { useSettings } from "./hooks/useSettings.ts";
import { useVisionStatus } from "./hooks/useVisionStatus.ts";
import { useVoice } from "./hooks/useVoice.ts";
import { useAppliedTheme } from "./theme.ts";

export default function App() {
  const [view, setView] = useState<"chat" | "settings">("chat");
  const settings = useSettings();
  // Created here, above the view switch, so the conversation survives a visit
  // to Settings.
  const chat = useChat();
  const voice = useVoice(chat);
  const memory = useMemoryStatus();
  const vision = useVisionStatus();
  // Treat "settings not loaded yet" as set so the hint does not flash on startup.
  const textModelSet =
    settings.view === null ||
    Boolean(settings.view.settings.models.text.path);
  const sttModelSet =
    settings.view === null ||
    Boolean(settings.view.settings.models.stt.path);
  useAppliedTheme(settings.view?.settings.appearance.theme ?? "system");

  return (
    <div className="flex h-full flex-col bg-background text-foreground text-sm">
      {view === "chat" ? (
        <>
          <AppHeader
            chat={chat}
            voice={voice}
            sttModelSet={sttModelSet}
            onOpenSettings={() => setView("settings")}
          />
          <ChatView
            controller={chat}
            textModelSet={textModelSet}
            onOpenSettings={() => setView("settings")}
            memory={memory}
            vision={vision}
            voiceNotice={voice.status?.error ?? voice.outputError}
          />
        </>
      ) : (
        <SettingsView
          onBack={() => setView("chat")}
          controller={settings}
        />
      )}
    </div>
  );
}
