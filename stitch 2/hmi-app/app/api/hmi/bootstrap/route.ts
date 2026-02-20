import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";

async function callFastApi<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${FASTAPI_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    cache: "no-store"
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`FastAPI ${path} failed: ${res.status} ${text}`);
  }

  return res.json() as Promise<T>;
}

export async function GET() {
  try {
    const [summary, correlations, samplePrediction] = await Promise.all([
      callFastApi("/dataset/summary"),
      callFastApi("/correlations/physical"),
      callFastApi("/predict/decay", {
        method: "POST",
        body: JSON.stringify({ ship_speed: 15, lever_pos: 5.1 })
      })
    ]);

    const recommendation = await callFastApi("/maintenance/recommend", {
      method: "POST",
      body: JSON.stringify({
        compressor_decay: (samplePrediction as { compressor_decay: number }).compressor_decay,
        turbine_decay: (samplePrediction as { turbine_decay: number }).turbine_decay
      })
    });

    return NextResponse.json({ summary, correlations, samplePrediction, recommendation });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Bootstrap failed",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    if (body.type === "predict") {
      const data = await callFastApi("/predict/decay", {
        method: "POST",
        body: JSON.stringify(body.payload)
      });
      return NextResponse.json(data);
    }

    if (body.type === "recommend") {
      const data = await callFastApi("/maintenance/recommend", {
        method: "POST",
        body: JSON.stringify(body.payload)
      });
      return NextResponse.json(data);
    }

    return NextResponse.json({ error: "Unsupported operation" }, { status: 400 });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Request failed",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}
