import { NextRequest, NextResponse } from "next/server";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";

export async function GET(req: NextRequest) {
  try {
    const speed = req.nextUrl.searchParams.get("speed");
    if (!speed) return NextResponse.json({ error: "speed query param required" }, { status: 400 });

    const res = await fetch(`${FASTAPI_BASE_URL}/hmi/surface-data?speed=${encodeURIComponent(speed)}`, {
      cache: "no-store"
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`FastAPI /hmi/surface-data failed: ${res.status} ${text}`);
    }
    return NextResponse.json(await res.json());
  } catch (error) {
    return NextResponse.json(
      {
        error: "Surface data fetch failed",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}
