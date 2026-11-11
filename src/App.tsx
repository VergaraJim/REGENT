import { useState } from "react";
import { AppHeader } from "./components/AppHeader.tsx";
import { ChatView } from "./views/ChatView.tsx";
import { SettingsView } from "./views/SettingsView.tsx";
import { useChat } from "./hooks/useChat.ts";
import { useSettings } from "./hooks/useSettings.ts";
import { useAppliedTheme } from "./theme.ts";

export default function App() {
  const [view, setView] = useState<"chat" | "settings">("chat");
  const settings = useSettings();
  // Created here, above the view switch, so the conversation survives a visit
  // to Settings.
  const chat = useChat();
  // Treat "settings not loaded yet" as set so the hint does not flash on startup.
  const textModelSet =
    settings.view === null ||
    Boolean(settings.view.settings.models.text.path);
  useAppliedTheme(settings.view?.settings.appearance.theme ?? "system");

  return (
    <div className="flex h-full flex-col bg-background text-foreground text-sm">
      {view === "chat" ? (
        <>
          <AppHeader
            chat={chat}
            onOpenSettings={() => setView("settings")}
          />
          <ChatView
            controller={chat}
            textModelSet={textModelSet}
            onOpenSettings={() => setView("settings")}
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
