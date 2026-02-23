import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const res = await fetch(`${FASTAPI_BASE_URL}/hmi/rul-prediction`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store"
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`FastAPI /hmi/rul-prediction failed: ${res.status} ${text}`);
    }
    return NextResponse.json(await res.json());
  } catch (error) {
    return NextResponse.json(
      {
        error: "RUL prediction fetch failed",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}
