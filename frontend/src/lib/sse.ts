// Minimal Server-Sent Events client for POST /chat.
// (The browser's EventSource only supports GET, so we read the fetch body ourselves.)

import { apiFetch, errorFrom, type Citation } from "./api";

export type ChatEvent =
  | { type: "start"; conversation_id: string; sources: number }
  | { type: "token"; text: string }
  | {
      type: "done";
      conversation_id: string;
      message_id: number;
      answer: string;
      citations: Citation[];
    }
  | { type: "error"; message: string };

export interface ChatRequest {
  question: string;
  conversation_id?: string;
  document_ids?: string[];
}

/** Parse complete SSE blocks from `buffer`; return the events and the unparsed remainder. */
export function parseSse(buffer: string): { events: ChatEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n?/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ChatEvent[] = [];
  for (const block of blocks) {
    let name = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) continue;
    try {
      events.push({ type: name, ...JSON.parse(data.join("\n")) } as ChatEvent);
    } catch {
      // Ignore malformed blocks rather than killing the stream.
    }
  }
  return { events, rest };
}

/** POST a question and call `onEvent` for each SSE event until the stream ends. */
export async function streamChat(
  body: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await apiFetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  // 429 (rate limited) and 404 (unknown conversation) carry a readable detail message.
  if (!res.ok || !res.body) throw await errorFrom(res);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const parsed = parseSse(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
  // Flush a final block that wasn't followed by a blank line.
  parseSse(buffer + "\n\n").events.forEach(onEvent);
}
