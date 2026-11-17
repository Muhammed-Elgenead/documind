import { describe, expect, it } from "vitest";

import { parseSse } from "./sse";

describe("parseSse", () => {
  it("parses complete events and keeps the partial remainder", () => {
    const input =
      'event: start\ndata: {"conversation_id": "c1", "sources": 8}\n\n' +
      'event: token\ndata: {"text": "Hel"}\n\n' +
      'event: token\ndata: {"te';
    const { events, rest } = parseSse(input);
    expect(events).toEqual([
      { type: "start", conversation_id: "c1", sources: 8 },
      { type: "token", text: "Hel" },
    ]);
    expect(rest).toBe('event: token\ndata: {"te');
  });

  it("reassembles events split across reads", () => {
    const first = parseSse('event: token\ndata: {"text": "a\\nb"}');
    expect(first.events).toEqual([]);
    const second = parseSse(first.rest + "\n\n");
    expect(second.events).toEqual([{ type: "token", text: "a\nb" }]);
  });

  it("handles CRLF line endings and skips malformed blocks", () => {
    const { events } = parseSse(
      "event: token\r\ndata: not json\r\n\r\nevent: error\r\ndata: {\"message\": \"x\"}\r\n\r\n",
    );
    expect(events).toEqual([{ type: "error", message: "x" }]);
  });
});
