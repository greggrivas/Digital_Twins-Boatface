import { mkdir, readFile, appendFile } from "fs/promises";
import { join } from "path";

const BASE_DIR = join(process.cwd(), "data");

export interface StoredMessage {
  sessionId: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  toolTrace?: string[];
}

async function ensureDataDir() {
  await mkdir(BASE_DIR, { recursive: true });
}

export async function appendJsonLine(fileName: string, entry: unknown) {
  await ensureDataDir();
  const path = join(BASE_DIR, fileName);
  await appendFile(path, `${JSON.stringify(entry)}\n`, "utf8");
}

export async function readJsonLines<T>(fileName: string): Promise<T[]> {
  await ensureDataDir();
  const path = join(BASE_DIR, fileName);

  try {
    const raw = await readFile(path, "utf8");
    return raw
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as T);
  } catch {
    return [];
  }
}

export async function listSessionMessages(sessionId: string): Promise<StoredMessage[]> {
  const all = await readJsonLines<StoredMessage>("chat_messages.jsonl");
  return all.filter((m) => m.sessionId === sessionId);
}
