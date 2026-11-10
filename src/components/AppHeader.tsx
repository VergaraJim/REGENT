import { Button } from "@heroui/react";
import { Settings } from "lucide-react";

export function AppHeader({ onOpenSettings }: { onOpenSettings: () => void }) {
  return (
    <header className="h-12 flex items-center border-b border-separator px-4">
      <span className="text-sm font-semibold">Regent</span>
      <div className="flex grow justify-end gap-2">
        <Button
          isIconOnly
          variant="ghost"
          size="sm"
          aria-label="Settings"
          onPress={onOpenSettings}
        >
          <Settings size={18} aria-hidden="true" />
        </Button>
      </div>
    </header>
  );
}
