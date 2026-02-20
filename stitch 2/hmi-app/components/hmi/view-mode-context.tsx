"use client";

import { createContext, useContext } from "react";

export type ViewMode = "dashboard" | "data";

export const ViewModeContext = createContext<ViewMode>("dashboard");

export function useViewMode() {
  return useContext(ViewModeContext);
}
