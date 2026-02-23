import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { appendJsonLine, listSessionMessages } from "@/lib/local-store";
import { readFile } from "fs/promises";
import { join } from "path";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL ?? "meta-llama/llama-3.1-70b-instruct";
const ASSISTANT_CONTEXT_PATH = join(process.cwd(), "data", "assistant_context.md");
const MAX_CONTEXT_CHARS = 8000;

let assistantContextCache: string | null = null;

const FALLBACK_ASSISTANT_CONTEXT = `
# Gas Turbine Digital Twin Context
- This assistant supports condition-based monitoring of a naval gas turbine digital twin.
- Do not invent numeric values or hidden assumptions. Use tool outputs as the source of truth.
- The snapshot endpoint uses a holdout-split CSV row for operating state and model-predicted decay.

## Sensor Notes
- TIC is Turbine Injection Control command (%) and is not direct fuel mass flow.
- Fuel_Flow (kg/s) is the actual fuel flow quantity.

## Health Bands
- Compressor: healthy >= 0.98, warning 0.96-0.98, critical < 0.96.
- Turbine: healthy >= 0.99, warning 0.98-0.99, critical < 0.98.

## RUL Notes
- RUL is reported for compressor and turbine separately.
- RUL units are dataset time-index units, not direct clock hours.

## Response Style
- Plain language, concise, operationally useful.
- Never expose tool names, function calls, JSON, or execution internals unless explicitly requested.
- If uncertain, state uncertainty directly.
`.trim();

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function textToHtml(text: string): string {
  const blocks = text
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter(Boolean);

  if (blocks.length === 0) return "<p>No response generated. Please rephrase your request.</p>";

  return blocks
    .map((block) => {
      const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
      const bulletLines = lines.filter((line) => line.startsWith("- "));
      const isPureBulletBlock = lines.length > 0 && bulletLines.length === lines.length;

      if (isPureBulletBlock) {
        const items = bulletLines
          .map((line) => `<li>${escapeHtml(line.slice(2).trim())}</li>`)
          .join("");
        return `<ul>${items}</ul>`;
      }

      return `<p>${escapeHtml(block).replaceAll("\n", "<br/>")}</p>`;
    })
    .join("");
}

function looksLikeHtml(text: string): boolean {
  return /<\/?[a-z][^>]*>/i.test(text);
}

function ensureHtmlContent(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return "<p>No response generated. Please rephrase your request.</p>";
  return looksLikeHtml(trimmed) ? trimmed : textToHtml(trimmed);
}

type SnapshotContext = {
  snapshot_id?: number;
  source?: string;
  operating_state?: {
    ship_speed?: number;
    lever_pos?: number;
    gt_torque?: number;
    gt_rpm?: number;
    gg_rpm?: number;
    tic?: number;
    fuel_flow?: number;
  };
  temperature_state?: {
    t48?: number;
    t2?: number;
    t1?: number;
  };
  pressure_state?: {
    p2?: number;
    p48?: number;
    p1?: number;
    pexh?: number;
  };
  predictions?: {
    compressor_decay_pred?: number;
    turbine_decay_pred?: number;
    severity?: string;
  };
  rul_prediction?: {
    compressor_rul_units?: number;
    turbine_rul_units?: number;
    next_component?: string;
    next_rul_units?: number;
  };
};

function buildSnapshotContextText(snapshot?: SnapshotContext | null): string | null {
  if (!snapshot?.operating_state) return null;
  const os = snapshot.operating_state;
  const ts = snapshot.temperature_state ?? {};
  const ps = snapshot.pressure_state ?? {};
  const pred = snapshot.predictions ?? {};
  const rul = snapshot.rul_prediction ?? {};

  return [
    `snapshot_id: ${snapshot.snapshot_id ?? "n/a"}`,
    `source: ${snapshot.source ?? "n/a"}`,
    `ship_speed_knots: ${os.ship_speed ?? "n/a"}`,
    `lever_pos: ${os.lever_pos ?? "n/a"}`,
    `gt_rpm: ${os.gt_rpm ?? "n/a"}`,
    `gg_rpm: ${os.gg_rpm ?? "n/a"}`,
    `gt_torque_knm: ${os.gt_torque ?? "n/a"}`,
    `tic_percent: ${os.tic ?? "n/a"}`,
    `fuel_flow_kgs: ${os.fuel_flow ?? "n/a"}`,
    `t48_c: ${ts.t48 ?? "n/a"}`,
    `t2_c: ${ts.t2 ?? "n/a"}`,
    `p2_bar: ${ps.p2 ?? "n/a"}`,
    `p48_bar: ${ps.p48 ?? "n/a"}`,
    `predicted_compressor_decay: ${pred.compressor_decay_pred ?? "n/a"}`,
    `predicted_turbine_decay: ${pred.turbine_decay_pred ?? "n/a"}`,
    `predicted_severity: ${pred.severity ?? "n/a"}`,
    `compressor_rul_units: ${rul.compressor_rul_units ?? "n/a"}`,
    `turbine_rul_units: ${rul.turbine_rul_units ?? "n/a"}`,
    `next_maintenance_component: ${rul.next_component ?? "n/a"}`,
    `next_maintenance_units: ${rul.next_rul_units ?? "n/a"}`
  ].join("\n");
}

function truncateContext(text: string, maxChars = MAX_CONTEXT_CHARS): string {
  return text.length <= maxChars ? text : `${text.slice(0, maxChars)}\n\n[Context truncated for token budget.]`;
}

async function loadAssistantContext(): Promise<string> {
  if (assistantContextCache) return assistantContextCache;
  try {
    const raw = await readFile(ASSISTANT_CONTEXT_PATH, "utf8");
    assistantContextCache = truncateContext(raw.trim());
    return assistantContextCache;
  } catch {
    assistantContextCache = FALLBACK_ASSISTANT_CONTEXT;
    return assistantContextCache;
  }
}

function formatNumber(value: unknown, digits = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return value.toFixed(digits);
}

function isToolishText(text: string): boolean {
  const t = text.toLowerCase();
  return (
    t.includes("function call") ||
    t.includes("tools:") ||
    t.includes("get_dataset_summary") ||
    t.includes("get_decay_prediction") ||
    t.includes("get_rul_prediction") ||
    t.includes("```json") ||
    (t.trim().startsWith("{") && t.includes("total_operating_points"))
  );
}

function normalizeToolResult(name: string, result: unknown): Record<string, unknown> {
  if (name === "get_current_snapshot") {
    return {
      tool: name,
      raw: result,
      explanation: {
        what_it_is:
          "Current operational snapshot where sensor values are sampled from a random dataset row, with decay states predicted by ML models.",
        caveats: ["Holdout sample, not a real-time ship sensor stream."],
        units: {
          ship_speed: "knots",
          fuel_flow: "kg/s",
          gt_torque: "kN·m",
          gt_rpm: "rpm",
          gg_rpm: "rpm"
        }
      }
    };
  }

  if (name === "get_dataset_summary") {
    return {
      tool: name,
      raw: result,
      explanation: {
        what_it_is: "Global statistics over the propulsion dataset used by the digital twin.",
        interpretation_guidance: [
          "critical_conditions and warning_conditions are counts of rows crossing configured decay bands.",
          "speed_range is the observed ship-speed span in the dataset."
        ]
      }
    };
  }

  if (name === "get_decay_prediction") {
    return {
      tool: name,
      raw: result,
      explanation: {
        what_it_is: "Model-predicted compressor and turbine decay coefficients for a given operating condition.",
        interpretation_guidance: [
          "Lower decay coefficients indicate worse component health.",
          "Severity is derived from configured decay thresholds."
        ]
      }
    };
  }

  if (name === "recommend_maintenance") {
    return {
      tool: name,
      raw: result,
      explanation: {
        what_it_is: "Maintenance recommendation derived from decay levels and severity logic.",
        interpretation_guidance: [
          "Action and priority indicate urgency.",
          "Maintenance window communicates recommended response timing."
        ]
      }
    };
  }

  if (name === "get_rul_prediction") {
    return {
      tool: name,
      raw: result,
      explanation: {
        what_it_is:
          "Linear remaining useful life projection in dataset time-index units for compressor and turbine.",
        interpretation_guidance: [
          "Lower RUL units means maintenance is needed sooner.",
          "Units are dataset progression units (CSV time-index), not direct clock hours."
        ]
      }
    };
  }

  return {
    tool: name,
    raw: result,
    explanation: {
      what_it_is: "Tool output for propulsion monitoring context.",
      caveats: ["Use values as reported; do not infer unavailable fields."]
    }
  };
}

function summarizeToolOutputs(toolOutputs: Record<string, unknown>[], toolTrace: string[]): string {
  const currentSnapshot = toolOutputs.find((t) => t.name === "get_current_snapshot")?.result as
    | {
        snapshot_id?: number;
        operating_state?: { ship_speed?: number; fuel_flow?: number };
        predictions?: {
          compressor_decay_pred?: number;
          turbine_decay_pred?: number;
          severity?: string;
        };
      }
    | undefined;

  const summary = toolOutputs.find((t) => t.name === "get_dataset_summary")?.result as
    | {
        total_operating_points?: number;
        speed_range?: [number, number];
        critical_conditions?: number;
        warning_conditions?: number;
        avg_compressor_decay?: number;
        avg_turbine_decay?: number;
      }
    | undefined;

  const prediction = toolOutputs.find((t) => t.name === "get_decay_prediction")?.result as
    | {
        compressor_decay?: number;
        turbine_decay?: number;
        severity?: string;
        operating_condition?: string;
      }
    | undefined;

  const recommendation = toolOutputs.find((t) => t.name === "recommend_maintenance")?.result as
    | {
        action?: string;
        priority?: string;
        components?: string[];
        maintenance_window?: string;
      }
    | undefined;

  const rul = toolOutputs.find((t) => t.name === "get_rul_prediction")?.result as
    | {
        compressor?: { rul_units?: number };
        turbine?: { rul_units?: number };
        next_maintenance?: { component?: string; rul_units?: number };
      }
    | undefined;

  const comparison = toolOutputs.find((t) => t.name === "compare_operating_conditions")?.result as
    | {
        speed_1_conditions?: { speed?: number; avg_T48?: number; avg_P48?: number };
        speed_2_conditions?: { speed?: number; avg_T48?: number; avg_P48?: number };
        recommendation?: string;
      }
    | undefined;

  if (currentSnapshot) {
    const lines = [
      `Current snapshot #${currentSnapshot.snapshot_id ?? "n/a"} (holdout CSV row):`,
      `- Ship speed: ${currentSnapshot.operating_state?.ship_speed ?? "n/a"} knots`,
      `- Fuel flow: ${formatNumber(currentSnapshot.operating_state?.fuel_flow, 3)} kg/s`,
      `- Predicted compressor decay: ${formatNumber(currentSnapshot.predictions?.compressor_decay_pred, 5)}`,
      `- Predicted turbine decay: ${formatNumber(currentSnapshot.predictions?.turbine_decay_pred, 5)}`,
      `- Health status: ${currentSnapshot.predictions?.severity ?? "n/a"}`
    ];

    if (rul) {
      lines.push(
        "",
        "RUL projection:",
        `- Compressor: ${rul.compressor?.rul_units ?? "n/a"} units`,
        `- Turbine: ${rul.turbine?.rul_units ?? "n/a"} units`,
        `- Next maintenance: ${rul.next_maintenance?.component ?? "n/a"} in ${rul.next_maintenance?.rul_units ?? "n/a"} units`
      );
    }
    return lines.join("\n");
  }

  if (summary) {
    return [
      "Dataset summary:",
      `- Total operating points: ${summary.total_operating_points ?? "n/a"}`,
      `- Speed range: ${summary.speed_range?.[0] ?? "n/a"} to ${summary.speed_range?.[1] ?? "n/a"} knots`,
      `- Critical conditions: ${summary.critical_conditions ?? "n/a"} | Warning conditions: ${summary.warning_conditions ?? "n/a"}`,
      `- Avg decay: compressor ${formatNumber(summary.avg_compressor_decay, 3)}, turbine ${formatNumber(summary.avg_turbine_decay, 4)}`
    ].join("\n");
  }

  if (prediction && recommendation) {
    const lines = [
      `At ${prediction.operating_condition ?? "the selected condition"}:`,
      `- Predicted compressor decay: ${formatNumber(prediction.compressor_decay, 5)}`,
      `- Predicted turbine decay: ${formatNumber(prediction.turbine_decay, 5)}`,
      `- Severity: ${prediction.severity ?? "n/a"}`,
      "",
      "Maintenance recommendation:",
      `- Action: ${recommendation.action ?? "n/a"} (${recommendation.priority ?? "n/a"} priority)`,
      `- Components: ${(recommendation.components ?? []).join(", ") || "n/a"}`,
      `- Window: ${recommendation.maintenance_window ?? "n/a"}`
    ];
    if (rul) {
      lines.push(
        "",
        "Predicted time to maintenance:",
        `- Compressor: ${rul.compressor?.rul_units ?? "n/a"} units`,
        `- Turbine: ${rul.turbine?.rul_units ?? "n/a"} units`,
        `- Limiting component: ${rul.next_maintenance?.component ?? "n/a"} (${rul.next_maintenance?.rul_units ?? "n/a"} units)`
      );
    }
    return lines.join("\n");
  }

  if (rul) {
    return [
      "Remaining useful life projection:",
      `- Compressor: ${rul.compressor?.rul_units ?? "n/a"} units`,
      `- Turbine: ${rul.turbine?.rul_units ?? "n/a"} units`,
      `- Next maintenance: ${rul.next_maintenance?.component ?? "n/a"} in ${rul.next_maintenance?.rul_units ?? "n/a"} units`
    ].join("\n");
  }

  if (prediction) {
    return [
      `Prediction at ${prediction.operating_condition ?? "selected condition"}:`,
      `- Compressor decay: ${formatNumber(prediction.compressor_decay, 5)}`,
      `- Turbine decay: ${formatNumber(prediction.turbine_decay, 5)}`,
      `- Severity: ${prediction.severity ?? "n/a"}`
    ].join("\n");
  }

  if (comparison) {
    return [
      "Operating condition comparison:",
      `- ${comparison.speed_1_conditions?.speed ?? "n/a"} knots: T48=${formatNumber(comparison.speed_1_conditions?.avg_T48, 2)}, P48=${formatNumber(comparison.speed_1_conditions?.avg_P48, 2)}`,
      `- ${comparison.speed_2_conditions?.speed ?? "n/a"} knots: T48=${formatNumber(comparison.speed_2_conditions?.avg_T48, 2)}, P48=${formatNumber(comparison.speed_2_conditions?.avg_P48, 2)}`,
      `- Recommendation: ${comparison.recommendation ?? "n/a"}`
    ].join("\n");
  }

  return `Tool execution complete (${toolTrace.join(", ")}).`;
}

async function callFastApi(path: string, body?: Record<string, unknown>, method = "POST") {
  const res = await fetch(`${FASTAPI_BASE_URL}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`FastAPI call failed: ${path} (${res.status}) ${text}`);
  }

  return res.json();
}

const tools: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_current_snapshot",
      description:
        "Get a current engine snapshot where operational values come from a holdout CSV row and decay values are model-predicted",
      parameters: {
        type: "object",
        properties: {}
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_decay_prediction",
      description: "Predict decay coefficients from ship speed and lever position",
      parameters: {
        type: "object",
        properties: {
          ship_speed: { type: "number" },
          lever_pos: { type: "number" }
        },
        required: ["ship_speed", "lever_pos"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_dataset_summary",
      description: "Get summary statistics of the local propulsion dataset",
      parameters: {
        type: "object",
        properties: {}
      }
    }
  },
  {
    type: "function",
    function: {
      name: "recommend_maintenance",
      description: "Get maintenance recommendation from decay coefficients",
      parameters: {
        type: "object",
        properties: {
          compressor_decay: { type: "number" },
          turbine_decay: { type: "number" }
        },
        required: ["compressor_decay", "turbine_decay"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_rul_prediction",
      description:
        "Get compressor and turbine remaining useful life (RUL) projection in units for the current or specified condition",
      parameters: {
        type: "object",
        properties: {
          ship_speed: { type: "number" },
          compressor_decay_pred: { type: "number" },
          turbine_decay_pred: { type: "number" },
          lever_pos: { type: "number" }
        }
      }
    }
  },
  {
    type: "function",
    function: {
      name: "get_physical_correlations",
      description: "Get sensor-decay physical correlations",
      parameters: {
        type: "object",
        properties: {}
      }
    }
  },
  {
    type: "function",
    function: {
      name: "compare_operating_conditions",
      description: "Compare two ship-speed operating conditions",
      parameters: {
        type: "object",
        properties: {
          speed_1: { type: "number" },
          speed_2: { type: "number" }
        },
        required: ["speed_1", "speed_2"]
      }
    }
  }
];

async function executeTool(
  toolName: string,
  args: Record<string, unknown>,
  snapshot?: SnapshotContext | null
) {
  if (toolName === "get_current_snapshot") {
    return callFastApi("/hmi/snapshot", undefined, "GET");
  }
  if (toolName === "get_decay_prediction") {
    const shipSpeedRaw = Number(args.ship_speed);
    const leverRaw = Number(args.lever_pos);
    const ship_speed = Number.isFinite(shipSpeedRaw) ? Math.min(27, Math.max(3, Math.round(shipSpeedRaw))) : 15;
    const lever_pos = Number.isFinite(leverRaw) ? Math.min(10, Math.max(1, leverRaw)) : 5.1;
    return callFastApi("/predict/decay", { ship_speed, lever_pos });
  }
  if (toolName === "get_dataset_summary") {
    return callFastApi("/dataset/summary", undefined, "GET");
  }
  if (toolName === "recommend_maintenance") {
    const compRaw = Number(args.compressor_decay);
    const turbRaw = Number(args.turbine_decay);

    let compressor_decay = compRaw;
    let turbine_decay = turbRaw;

    const invalidComp = !Number.isFinite(compressor_decay) || compressor_decay < 0.9 || compressor_decay > 1.0;
    const invalidTurb = !Number.isFinite(turbine_decay) || turbine_decay < 0.9 || turbine_decay > 1.0;

    if (invalidComp || invalidTurb) {
      const shipSpeedRaw = Number(args.ship_speed);
      const leverRaw = Number(args.lever_pos);
      const ship_speed = Number.isFinite(shipSpeedRaw) ? Math.min(27, Math.max(3, Math.round(shipSpeedRaw))) : 15;
      const lever_pos = Number.isFinite(leverRaw) ? Math.min(10, Math.max(1, leverRaw)) : 5.1;

      const prediction = (await callFastApi("/predict/decay", {
        ship_speed,
        lever_pos
      })) as { compressor_decay: number; turbine_decay: number };
      compressor_decay = prediction.compressor_decay;
      turbine_decay = prediction.turbine_decay;
    }

    return callFastApi("/maintenance/recommend", {
      compressor_decay,
      turbine_decay
    });
  }
  if (toolName === "get_rul_prediction") {
    const os = snapshot?.operating_state ?? {};
    const pred = snapshot?.predictions ?? {};

    const shipSpeedRaw = Number(args.ship_speed ?? os.ship_speed);
    const leverRaw = Number(args.lever_pos ?? os.lever_pos);
    const ship_speed = Number.isFinite(shipSpeedRaw)
      ? Math.min(27, Math.max(3, Math.round(shipSpeedRaw)))
      : 15;
    const lever_pos = Number.isFinite(leverRaw)
      ? Math.min(10, Math.max(1, leverRaw))
      : 5.1;

    const compRaw = Number(
      args.compressor_decay_pred ?? args.compressor_decay ?? pred.compressor_decay_pred
    );
    const turbRaw = Number(
      args.turbine_decay_pred ?? args.turbine_decay ?? pred.turbine_decay_pred
    );

    let compressor_decay_pred = compRaw;
    let turbine_decay_pred = turbRaw;

    const invalidComp =
      !Number.isFinite(compressor_decay_pred) ||
      compressor_decay_pred < 0.9 ||
      compressor_decay_pred > 1.0;
    const invalidTurb =
      !Number.isFinite(turbine_decay_pred) ||
      turbine_decay_pred < 0.9 ||
      turbine_decay_pred > 1.0;

    if (invalidComp || invalidTurb) {
      const prediction = (await callFastApi("/predict/decay", {
        ship_speed,
        lever_pos
      })) as { compressor_decay: number; turbine_decay: number };
      compressor_decay_pred = prediction.compressor_decay;
      turbine_decay_pred = prediction.turbine_decay;
    }

    return callFastApi("/hmi/rul-prediction", {
      ship_speed,
      compressor_decay_pred,
      turbine_decay_pred
    });
  }
  if (toolName === "get_physical_correlations") {
    return callFastApi("/correlations/physical", undefined, "GET");
  }
  if (toolName === "compare_operating_conditions") {
    const s1Raw = Number(args.speed_1);
    const s2Raw = Number(args.speed_2);
    const speed_1 = Number.isFinite(s1Raw) ? Math.min(27, Math.max(3, Math.round(s1Raw))) : 15;
    const speed_2 = Number.isFinite(s2Raw) ? Math.min(27, Math.max(3, Math.round(s2Raw))) : 27;
    return callFastApi("/tools/compare-operating-conditions", { speed_1, speed_2 });
  }
  throw new Error(`Unknown tool: ${toolName}`);
}

export async function POST(req: NextRequest) {
  try {
    if (!process.env.OPENROUTER_API_KEY) {
      return NextResponse.json({ error: "OPENROUTER_API_KEY is missing" }, { status: 500 });
    }

    const { sessionId, message, currentSnapshot } = (await req.json()) as {
      sessionId: string;
      message: string;
      currentSnapshot?: SnapshotContext | null;
    };
    const snapshotContextText = buildSnapshotContextText(currentSnapshot);

    const client = new OpenAI({
      apiKey: process.env.OPENROUTER_API_KEY,
      baseURL: "https://openrouter.ai/api/v1"
    });

    const assistantContext = await loadAssistantContext();
    const priorMessages = await listSessionMessages(sessionId);

    const conversation: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: [
          "You are a marine propulsion maintenance assistant.",
          "Never invent numeric values.",
          "Always use tools for predictions or statistics when available.",
          "For any question about predicted time to maintenance or RUL, call get_rul_prediction.",
          "This is condition monitoring, not time-based prognostics.",
          "Respond in plain, human-readable language as an HTML fragment.",
          "Allowed tags: <p>, <ul>, <ol>, <li>, <strong>, <em>, <br>, <code>, <pre>, <a>.",
          "Do not output markdown fences, and do not output <html>, <head>, <body>, <script>, or inline event handlers.",
          "Do not mention function names, tool calls, JSON blocks, or internal execution steps unless the user explicitly asks for raw output.",
          snapshotContextText
            ? "If a Current HMI Snapshot is provided below, treat it as the authoritative current state for this turn."
            : "",
          snapshotContextText
            ? "Do not call get_current_snapshot unless the user explicitly asks to refresh/reload the snapshot."
            : "",
          "",
          "Project context:",
          assistantContext,
          snapshotContextText ? "" : "",
          snapshotContextText ? "Current HMI Snapshot (authoritative):" : "",
          snapshotContextText ?? ""
        ].join("\n")
      },
      ...priorMessages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user", content: message }
    ];

    await appendJsonLine("chat_messages.jsonl", {
      sessionId,
      role: "user",
      content: message,
      createdAt: new Date().toISOString()
    });

    const availableTools =
      snapshotContextText != null
        ? tools.filter((t) => t.type === "function" && t.function.name !== "get_current_snapshot")
        : tools;

    const first = await client.chat.completions.create({
      model: OPENROUTER_MODEL,
      messages: conversation,
      tools: availableTools,
      tool_choice: "auto"
    });

    const toolTrace: string[] = [];
    const toolOutputs: Record<string, unknown>[] = [];
    const choice = first.choices[0];
    const assistantMessage = choice?.message;

    if (assistantMessage?.tool_calls?.length) {
      conversation.push(assistantMessage);

      for (const toolCall of assistantMessage.tool_calls) {
        const name = toolCall.function.name;
        const parsedArgs = JSON.parse(toolCall.function.arguments || "{}");
        const toolResult = await executeTool(name, parsedArgs, currentSnapshot);
        toolTrace.push(name);
        toolOutputs.push({ name, result: toolResult });

        conversation.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: JSON.stringify(toolResult)
        });
      }

      const normalizedToolOutputs = toolOutputs.map((entry) => {
        const name = String(entry.name ?? "");
        return normalizeToolResult(name, entry.result);
      });

      const summarizerMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
        {
          role: "system",
          content: [
            "You summarize propulsion tool outputs for an HMI operator.",
            "Write concise, user-facing explanations with context and operational implications.",
            "Output HTML fragments only (no markdown).",
            "Allowed tags: <p>, <ul>, <ol>, <li>, <strong>, <em>, <br>, <code>, <pre>, <a>.",
            "Do not output <html>, <head>, <body>, <script>, style attributes, or inline event handlers.",
            "Never mention tool names, function calls, JSON, execution traces, or internal mechanics.",
            "Never invent numbers; use only provided data.",
            "If data is missing, say what is missing and what can still be concluded.",
            "If RUL data is available, report compressor and turbine RUL explicitly in units.",
            "",
            "Project context:",
            assistantContext
          ].join("\n")
        },
        {
          role: "user",
          content: [
            `User request: ${message}`,
            "",
            snapshotContextText ? "Authoritative current HMI snapshot for this turn:" : "",
            snapshotContextText ?? "",
            snapshotContextText ? "" : "",
            "Normalized tool outputs:",
            JSON.stringify(normalizedToolOutputs, null, 2),
            "",
            "Produce an operator-facing HTML fragment."
          ].join("\n")
        }
      ];

      const second = await client.chat.completions.create({
        model: OPENROUTER_MODEL,
        messages: summarizerMessages
      });

      const modelSummary = (second.choices[0]?.message?.content ?? "").trim();
      const fallbackSummary = summarizeToolOutputs(toolOutputs, toolTrace);
      const finalTextRaw =
        modelSummary.length > 0 && !isToolishText(modelSummary)
          ? modelSummary
          : fallbackSummary;
      const finalText = ensureHtmlContent(finalTextRaw);

      const response = {
        id: `a-${crypto.randomUUID()}`,
        role: "assistant" as const,
        content: finalText,
        createdAt: new Date().toISOString(),
        toolTrace
      };

      await appendJsonLine("chat_messages.jsonl", { sessionId, ...response });
      return NextResponse.json(response);
    }

    const rawText = assistantMessage?.content ?? "";
    const finalTextRaw =
      typeof rawText === "string" && rawText.trim().length > 0
        ? rawText
        : "No textual response generated. Please rephrase your request.";
    const finalText = ensureHtmlContent(finalTextRaw);
    const response = {
      id: `a-${crypto.randomUUID()}`,
      role: "assistant" as const,
      content: finalText,
      createdAt: new Date().toISOString(),
      toolTrace
    };

    await appendJsonLine("chat_messages.jsonl", { sessionId, ...response });
    return NextResponse.json(response);
  } catch (error) {
    return NextResponse.json(
      {
        error: "Chat failed",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}
