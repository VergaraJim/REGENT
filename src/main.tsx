import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/inter";
import "./index.css";
import { applySystemThemeBeforeFirstRender } from "./theme.ts";
import App from "./App.tsx";

applySystemThemeBeforeFirstRender();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
