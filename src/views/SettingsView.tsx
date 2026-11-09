import { Button } from "@heroui/react";
import { ChevronLeft } from "lucide-react";

export function SettingsView({ onBack }: { onBack: () => void }) {
  return (
    <>
      <header className="h-12 flex items-center border-b border-divider px-4">
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
      <main className="p-4">
        <p className="text-sm text-default-500">Settings will appear here.</p>
      </main>
    </>
  );
}
