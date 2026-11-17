import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { apiFetch, ApiError, errorFrom, getToken, setToken, UNAUTHORIZED_EVENT } from "./api";

function memoryStorage() {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
  };
}

describe("apiFetch", () => {
  let events: string[];

  beforeEach(() => {
    events = [];
    vi.stubGlobal("localStorage", memoryStorage());
    vi.stubGlobal("window", { dispatchEvent: (e: Event) => events.push(e.type) });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("sends the bearer token when signed in", async () => {
    const fetchMock = vi.fn(async () => new Response("[]", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    setToken("abc");

    await apiFetch("/documents", { headers: { Accept: "application/json" } });

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/documents");
    const headers = new Headers(init.headers);
    expect(headers.get("Authorization")).toBe("Bearer abc");
    expect(headers.get("Accept")).toBe("application/json");
  });

  it("sends no header when signed out", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await apiFetch("/health");

    const init = (fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1];
    expect(new Headers(init.headers).has("Authorization")).toBe(false);
  });

  it("clears the session and notifies the app on 401", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 401 })));
    setToken("expired");

    const res = await apiFetch("/documents");

    expect(res.status).toBe(401);
    expect(getToken()).toBeNull();
    expect(events).toEqual([UNAUTHORIZED_EVENT]);
  });
});

describe("errorFrom", () => {
  it("uses the API's detail message, including validation errors", async () => {
    const tooMany = new Response(JSON.stringify({ detail: "Too many requests. Please try again in 12 s." }), {
      status: 429,
    });
    const err = await errorFrom(tooMany);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(429);
    expect(err.message).toBe("Too many requests. Please try again in 12 s.");

    const invalid = new Response(JSON.stringify({ detail: [{ msg: "value is not a valid email address" }] }), {
      status: 422,
    });
    expect((await errorFrom(invalid)).message).toBe("value is not a valid email address");
  });
});
