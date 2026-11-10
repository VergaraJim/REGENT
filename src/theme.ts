import { useEffect, useState } from "react";

export type ThemePreference = "system" | "light" | "dark";

export type ResolvedTheme = "light" | "dark";

/** Pure: maps a preference plus the system's dark setting to a concrete theme. */
export function resolveTheme(
  pref: ThemePreference,
  systemPrefersDark: boolean,
): ResolvedTheme {
  if (pref === "system") return systemPrefersDark ? "dark" : "light";
  return pref;
}

/** Applies a resolved theme to the document (toggles `dark` and sets color-scheme). */
export function applyTheme(resolved: ResolvedTheme): void {
  const root = document.documentElement;
  root.classList.toggle("dark", resolved === "dark");
  root.style.colorScheme = resolved;
}

/** Watches `prefers-color-scheme` and keeps the applied theme in sync. */
export function useAppliedTheme(pref: ThemePreference): ResolvedTheme {
  const [applied, setApplied] = useState<ResolvedTheme>(() =>
    resolveTheme(pref, window.matchMedia("(prefers-color-scheme: dark)").matches),
  );

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => {
      const resolved = resolveTheme(pref, media.matches);
      setApplied(resolved);
      applyTheme(resolved);
    };
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [pref]);

  return applied;
}

/** Called once before the first render so the system theme does not flash. */
export function applySystemThemeBeforeFirstRender(): void {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  applyTheme(resolveTheme("system", media.matches));
}
