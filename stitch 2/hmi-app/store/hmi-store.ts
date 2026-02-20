import { create } from "zustand";
import type { ChatMessage } from "@/lib/types";

interface HmiState {
  selectedUnit: string;
  timeWindow: "24h" | "7d" | "30d";
  sessionId: string;
  messages: ChatMessage[];
  setUnit: (unit: string) => void;
  setTimeWindow: (w: "24h" | "7d" | "30d") => void;
  addMessage: (msg: ChatMessage) => void;
  addLocalUserMessage: (content: string) => void;
}

function generateSessionId() {
  return `session-${Math.random().toString(36).slice(2, 10)}`;
}

export const useHmiStore = create<HmiState>((set) => ({
  selectedUnit: "Propulsion Unit 04",
  timeWindow: "24h",
  sessionId: generateSessionId(),
  messages: [],
  setUnit: (unit) => set({ selectedUnit: unit }),
  setTimeWindow: (timeWindow) => set({ timeWindow }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  addLocalUserMessage: (content) =>
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id: `u-${crypto.randomUUID()}`,
          role: "user",
          content,
          createdAt: new Date().toISOString()
        }
      ]
    }))
}));
