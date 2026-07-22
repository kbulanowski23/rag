import type { ChatTurn, EffectiveConfig, IndexStats, Source } from "./types";

// Resolved at build time from NEXT_PUBLIC_API_BASE_URL. In OpenShift this is set
// on the build, not at runtime, because Next inlines NEXT_PUBLIC_* values.
// If the API is exposed on the same route as the UI, leave it empty and use
// relative paths.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
const V1 = `${API_BASE}/api/v1`;

export type StreamEvent =
  | { type: "sources"; sources: Source[]; retrieval_ms: number }
  | { type: "token"; text: string }
  | { type: "done"; cited: number[]; timings_ms: Record<string, number> }
  | { type: "error"; message: string };

/**
 * Streams a chat answer over SSE.
 *
 * EventSource cannot be used here: it only issues GET requests and cannot send
 * a JSON body. So we POST with fetch and parse the SSE framing off the response
 * stream ourselves -- which is about twenty lines and avoids a dependency.
 */
export async function streamChat(
  question: string,
  opts: {
    history?: ChatTurn[];
    k?: number;
    filters?: Record<string, unknown>;
    signal?: AbortSignal;
    onEvent: (event: StreamEvent) => void;
  },
): Promise<void> {
  const response = await fetch(`${V1}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      history: opts.history ?? [],
      k: opts.k,
      filters: opts.filters,
      stream: true,
    }),
    signal: opts.signal,
  });

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(`chat failed (${response.status}): ${detail.slice(0, 300)}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. A chunk boundary can fall
    // mid-frame, so anything after the last separator stays in the buffer.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        try {
          opts.onEvent(JSON.parse(line.slice(5).trim()) as StreamEvent);
        } catch {
          // Ignore keepalives and any frame we do not recognise.
        }
      }
    }
  }
}

export async function search(query: string, k?: number) {
  const r = await fetch(`${V1}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });
  if (!r.ok) throw new Error(`search failed (${r.status})`);
  return (await r.json()) as { query: string; hits: Source[]; took_ms: number };
}

export async function uploadDocument(file: File, metadata?: Record<string, unknown>) {
  const form = new FormData();
  form.append("file", file);
  form.append("source_uri", file.name);
  if (metadata) form.append("metadata", JSON.stringify(metadata));

  const r = await fetch(`${V1}/ingest`, { method: "POST", body: form });
  if (!r.ok) {
    const detail = await r.text().catch(() => r.statusText);
    throw new Error(`upload failed (${r.status}): ${detail.slice(0, 300)}`);
  }
  return r.json();
}

export async function getConfig(): Promise<EffectiveConfig> {
  const r = await fetch(`${V1}/config`, { cache: "no-store" });
  if (!r.ok) throw new Error(`config failed (${r.status})`);
  return r.json();
}

export async function getIndexStats(): Promise<IndexStats> {
  const r = await fetch(`${V1}/index/stats`, { cache: "no-store" });
  if (!r.ok) throw new Error(`stats failed (${r.status})`);
  return r.json();
}
