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
- The snapshot endpoint uses a random CSV row for operating state and model-predicted decay.

## Sensor Notes
- TIC is Turbine Injection Control command (%) and is not direct fuel mass flow.
- Fuel_Flow (kg/s) is the actual fuel flow quantity.

## Health Bands
- Compressor: healthy >= 0.98, warning 0.96-0.98, critical < 0.96.
- Turbine: healthy >= 0.99, warning 0.98-0.99, critical < 0.98.

## Response Style
- Plain language, concise, operationally useful.
- Never expose tool names, function calls, JSON, or execution internals unless explicitly requested.
- If uncertain, state uncertainty directly.
`.trim();

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
        caveats: ["Random sample, not a real-time ship sensor stream."],
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

  const comparison = toolOutputs.find((t) => t.name === "compare_operating_conditions")?.result as
    | {
        speed_1_conditions?: { speed?: number; avg_T48?: number; avg_P48?: number };
        speed_2_conditions?: { speed?: number; avg_T48?: number; avg_P48?: number };
        recommendation?: string;
      }
    | undefined;

  if (currentSnapshot) {
    return [
      `Current snapshot #${currentSnapshot.snapshot_id ?? "n/a"} (random CSV row):`,
      `- Ship speed: ${currentSnapshot.operating_state?.ship_speed ?? "n/a"} knots`,
      `- Fuel flow: ${formatNumber(currentSnapshot.operating_state?.fuel_flow, 3)} kg/s`,
      `- Predicted compressor decay: ${formatNumber(currentSnapshot.predictions?.compressor_decay_pred, 5)}`,
      `- Predicted turbine decay: ${formatNumber(currentSnapshot.predictions?.turbine_decay_pred, 5)}`,
      `- Health status: ${currentSnapshot.predictions?.severity ?? "n/a"}`
    ].join("\n");
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
    return [
      `At ${prediction.operating_condition ?? "the selected condition"}:`,
      `- Predicted compressor decay: ${formatNumber(prediction.compressor_decay, 5)}`,
      `- Predicted turbine decay: ${formatNumber(prediction.turbine_decay, 5)}`,
      `- Severity: ${prediction.severity ?? "n/a"}`,
      "",
      "Maintenance recommendation:",
      `- Action: ${recommendation.action ?? "n/a"} (${recommendation.priority ?? "n/a"} priority)`,
      `- Components: ${(recommendation.components ?? []).join(", ") || "n/a"}`,
      `- Window: ${recommendation.maintenance_window ?? "n/a"}`
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
        "Get a current engine snapshot where operational values come from a random CSV row and decay values are model-predicted",
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

async function executeTool(toolName: string, args: Record<string, unknown>) {
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

    const { sessionId, message } = (await req.json()) as {
      sessionId: string;
      message: string;
    };

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
          "This is condition monitoring, not time-based prognostics.",
          "Respond in plain, human-readable language.",
          "Do not mention function names, tool calls, JSON blocks, or internal execution steps unless the user explicitly asks for raw output.",
          "",
          "Project context:",
          assistantContext
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

    const first = await client.chat.completions.create({
      model: OPENROUTER_MODEL,
      messages: conversation,
      tools,
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
        const toolResult = await executeTool(name, parsedArgs);
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
            "Never mention tool names, function calls, JSON, execution traces, or internal mechanics.",
            "Never invent numbers; use only provided data.",
            "If data is missing, say what is missing and what can still be concluded.",
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
            "Normalized tool outputs:",
            JSON.stringify(normalizedToolOutputs, null, 2),
            "",
            "Produce a plain-language answer suitable for operators."
          ].join("\n")
        }
      ];

      const second = await client.chat.completions.create({
        model: OPENROUTER_MODEL,
        messages: summarizerMessages
      });

      const modelSummary = (second.choices[0]?.message?.content ?? "").trim();
      const fallbackSummary = summarizeToolOutputs(toolOutputs, toolTrace);
      const finalText =
        modelSummary.length > 0 && !isToolishText(modelSummary)
          ? modelSummary
          : fallbackSummary;

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
    const finalText =
      typeof rawText === "string" && rawText.trim().length > 0
        ? rawText
        : "No textual response generated. Please rephrase your request.";
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
