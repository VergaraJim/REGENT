import { useState } from "react";
import { AppHeader } from "./components/AppHeader.tsx";
import { ChatView } from "./views/ChatView.tsx";
import { SettingsView } from "./views/SettingsView.tsx";
import { useAppliedTheme } from "./theme.ts";

export default function App() {
  const [view, setView] = useState<"chat" | "settings">("chat");
  useAppliedTheme("system");

  return (
    <div className="flex h-full flex-col bg-background text-foreground text-sm">
      {view === "chat" ? (
        <>
          <AppHeader onOpenSettings={() => setView("settings")} />
          <ChatView />
        </>
      ) : (
        <SettingsView onBack={() => setView("chat")} />
      )}
    </div>
  );
}
