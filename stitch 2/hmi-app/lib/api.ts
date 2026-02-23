import type {
  BootstrapPayload,
  ChatMessage,
  DecayPrediction,
  HmiRulPrediction,
  HmiSnapshot,
  HmiSurfaceData,
  HmiSurfaceMarker,
  MaintenanceRecommendation
} from "@/lib/types";

export async function fetchBootstrap(): Promise<BootstrapPayload> {
  const res = await fetch("/api/hmi/bootstrap", { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load dashboard bootstrap data");
  return res.json();
}

export async function predictDecay(input: {
  ship_speed: number;
  lever_pos: number;
}): Promise<DecayPrediction> {
  const res = await fetch("/api/hmi/bootstrap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: "predict", payload: input })
  });

  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}

export async function recommendMaintenance(input: {
  compressor_decay: number;
  turbine_decay: number;
}): Promise<MaintenanceRecommendation> {
  const res = await fetch("/api/hmi/bootstrap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: "recommend", payload: input })
  });

  if (!res.ok) throw new Error("Recommendation request failed");
  return res.json();
}

export async function sendChatMessage(input: {
  sessionId: string;
  message: string;
  currentSnapshot?: HmiSnapshot | null;
}): Promise<ChatMessage> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input)
  });

  if (!res.ok) throw new Error("Chat request failed");
  return res.json();
}

export async function fetchHmiSnapshot(): Promise<HmiSnapshot> {
  const res = await fetch("/api/hmi/snapshot", { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load HMI snapshot");
  return res.json();
}

export async function fetchSurfaceData(speed: number): Promise<HmiSurfaceData> {
  const res = await fetch(`/api/hmi/surface-data?speed=${encodeURIComponent(speed)}`, {
    cache: "no-store"
  });
  if (!res.ok) throw new Error("Failed to load surface data");
  return res.json();
}

export async function fetchSurfaceMarker(input: {
  speed: number;
  compressor_decay_pred: number;
  turbine_decay_pred: number;
}): Promise<HmiSurfaceMarker> {
  const res = await fetch("/api/hmi/surface-marker", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input)
  });
  if (!res.ok) throw new Error("Failed to load surface marker");
  return res.json();
}

export async function fetchRulPrediction(input: {
  ship_speed: number;
  compressor_decay_pred: number;
  turbine_decay_pred: number;
}): Promise<HmiRulPrediction> {
  const res = await fetch("/api/hmi/rul-prediction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input)
  });
  if (!res.ok) throw new Error("Failed to load RUL prediction");
  return res.json();
}
