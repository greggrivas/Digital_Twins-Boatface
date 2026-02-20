import { NextResponse } from "next/server";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const res = await fetch(`${FASTAPI_BASE_URL}/hmi/snapshot`, { cache: "no-store" });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`FastAPI /hmi/snapshot failed: ${res.status} ${text}`);
    }
    return NextResponse.json(await res.json());
  } catch (error) {
    return NextResponse.json(
      {
        error: "Snapshot fetch failed",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}
