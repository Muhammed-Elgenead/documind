// Typed client for the FastAPI backend.
// In development calls go through the Vite proxy at /api. In production VITE_API_URL
// points at the deployed backend (e.g. https://documind-api.up.railway.app).

export type DocumentStatus = "processing" | "ready" | "failed";

export interface DocumentItem {
  id: string;
  filename: string;
  status: DocumentStatus;
  page_count: number | null;
  error: string | null;
  created_at: string;
}

export interface Citation {
  number: number;
  document_id: string;
  filename: string;
  page: number;
  chunk_id: number;
  text: string;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
}

export interface Conversation extends ConversationSummary {
  messages: ChatMessage[];
}

export type Health = {
  status: "healthy" | "unhealthy";
  database: "ok" | "error";
};

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: "bearer";
  user: User;
}

export const API = (import.meta.env.VITE_API_URL ?? "/api").replace(/\/+$/, "");

export const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".md", ".markdown"];
export const MAX_UPLOAD_MB = 50;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

// ---------- Session token ----------
// A bearer token in localStorage: simple, and works when the frontend and API are on
// different domains (cookies would need SameSite=None and break in browsers that block
// third-party cookies). The trade-off: any XSS could read it, so the UI never renders
// untrusted HTML (react-markdown escapes it).

const TOKEN_KEY = "documind.token";
export const UNAUTHORIZED_EVENT = "documind:unauthorized";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable (private mode): the session lasts until reload */
  }
}

export async function errorFrom(res: Response): Promise<ApiError> {
  let message = `Request failed (HTTP ${res.status})`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") message = body.detail;
    else if (Array.isArray(body?.detail) && typeof body.detail[0]?.msg === "string") {
      message = body.detail[0].msg; // FastAPI validation error
    }
  } catch {
    /* not JSON */
  }
  return new ApiError(message, res.status);
}

/** fetch() against the API with the session token; a 401 signs the user out everywhere. */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(API + path, { ...init, headers });
  if (res.status === 401 && token) {
    setToken(null);
    window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
  }
  return res;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await apiFetch(path);
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as T;
}

// ---------- Endpoints ----------

export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${API}/health`);
  // 503 still carries a Health body describing what is down.
  if (res.ok || res.status === 503) return (await res.json()) as Health;
  throw await errorFrom(res);
}

export const register = (email: string, password: string) =>
  postJson<AuthResponse>("/auth/register", { email, password });
export const login = (email: string, password: string) =>
  postJson<AuthResponse>("/auth/login", { email, password });
export const fetchMe = () => getJson<User>("/auth/me");

export const listDocuments = () => getJson<DocumentItem[]>("/documents");
export const listConversations = () => getJson<ConversationSummary[]>("/conversations");
export const getConversation = (id: string) => getJson<Conversation>(`/conversations/${id}`);

export async function uploadDocument(file: File): Promise<DocumentItem> {
  const form = new FormData();
  form.append("file", file);
  const res = await apiFetch("/documents", { method: "POST", body: form });
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as DocumentItem;
}

/** Client-side check before uploading; the server validates again. */
export function validateFile(file: File): string | null {
  const name = file.name.toLowerCase();
  if (!ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return "Unsupported type. Upload PDF, DOCX or Markdown.";
  }
  if (file.size === 0) return "File is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `File exceeds ${MAX_UPLOAD_MB} MB.`;
  return null;
}
