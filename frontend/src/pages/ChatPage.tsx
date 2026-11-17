import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertCircle, ArrowUp, FileText, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { CitationPanel } from "../components/CitationPanel";
import { Markdown } from "../components/Markdown";
import { type ChatMessage, type Citation, getConversation, listDocuments } from "../lib/api";
import { streamChat } from "../lib/sse";

interface UiMessage {
  key: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  streaming?: boolean;
  error?: string;
}

const toUi = (m: ChatMessage): UiMessage => ({
  key: String(m.id),
  role: m.role,
  content: m.content,
  citations: m.citations,
});

const SUGGESTIONS = [
  "Summarize the key points",
  "What are the main risks mentioned?",
  "List the important numbers and dates",
];

export function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);

  // The conversation whose messages are already in local state (loaded or being streamed).
  const loadedFor = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Switching conversations resets the view, unless the URL changed because
  // our own stream just created this conversation.
  useEffect(() => {
    if (conversationId && conversationId === loadedFor.current) return;
    abortRef.current?.abort();
    loadedFor.current = null;
    setMessages([]);
    setOpenCitation(null);
  }, [conversationId]);

  const conversation = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => getConversation(conversationId!),
    enabled: !!conversationId && loadedFor.current !== conversationId,
    retry: false,
  });

  useEffect(() => {
    const data = conversation.data;
    if (data && data.id === conversationId && loadedFor.current !== conversationId) {
      loadedFor.current = conversationId;
      setMessages(data.messages.map(toUi));
    }
  }, [conversation.data, conversationId]);

  const documents = useQuery({ queryKey: ["documents"], queryFn: listDocuments });
  const readyCount = documents.data?.filter((d) => d.status === "ready").length ?? 0;

  // Keep the newest content in view.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const send = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || streaming) return;

      const assistantKey = crypto.randomUUID();
      const update = (fn: (m: UiMessage) => UiMessage) =>
        setMessages((all) => all.map((m) => (m.key === assistantKey ? fn(m) : m)));

      setMessages((all) => [
        ...all,
        { key: crypto.randomUUID(), role: "user", content: q, citations: null },
        { key: assistantKey, role: "assistant", content: "", citations: null, streaming: true },
      ]);
      setStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamChat(
          { question: q, conversation_id: conversationId },
          (event) => {
            switch (event.type) {
              case "start":
                if (!conversationId) {
                  loadedFor.current = event.conversation_id;
                  navigate(`/c/${event.conversation_id}`, { replace: true });
                }
                break;
              case "token":
                update((m) => ({ ...m, content: m.content + event.text }));
                break;
              case "done":
                update((m) => ({
                  ...m,
                  content: event.answer,
                  citations: event.citations,
                  streaming: false,
                }));
                queryClient.removeQueries({ queryKey: ["conversation", event.conversation_id] });
                queryClient.invalidateQueries({ queryKey: ["conversations"] });
                break;
              case "error":
                update((m) => ({ ...m, streaming: false, error: event.message }));
                break;
            }
          },
          controller.signal,
        );
      } catch (err) {
        const aborted = controller.signal.aborted;
        update((m) => ({
          ...m,
          streaming: false,
          error: aborted ? (m.content ? undefined : "Stopped.") : (err as Error).message,
        }));
      } finally {
        // Stream ended without "done" or "error" (e.g. connection dropped).
        update((m) =>
          m.streaming ? { ...m, streaming: false, error: "The answer was interrupted." } : m,
        );
        if (abortRef.current === controller) abortRef.current = null;
        setStreaming(false);
      }
    },
    [conversationId, navigate, queryClient, streaming],
  );

  const closePanel = useCallback(() => setOpenCitation(null), []);
  const empty = messages.length === 0 && !conversationId;

  return (
    <div className="flex min-w-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-3xl px-4 py-6 sm:px-6">
            {conversation.isError && (
              <p className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700">
                Conversation not found. <Link to="/" className="underline">Start a new chat</Link>.
              </p>
            )}
            {conversationId && conversation.isPending && messages.length === 0 && (
              <p className="text-sm text-slate-400">Loading conversation…</p>
            )}

            {empty && (
              <div className="pt-[12vh] text-center">
                <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                  Ask your documents
                </h1>
                <p className="mt-2 text-slate-500">
                  {readyCount > 0 ? (
                    <>Answers come only from your {readyCount} ready document{readyCount === 1 ? "" : "s"}, with page citations.</>
                  ) : (
                    <>
                      No documents yet.{" "}
                      <Link to="/documents" className="font-medium text-indigo-600 underline">
                        Upload one
                      </Link>{" "}
                      to get started.
                    </>
                  )}
                </p>
                {readyCount > 0 && (
                  <div className="mt-6 flex flex-wrap justify-center gap-2">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => send(s)}
                        className="rounded-full border border-slate-200 px-3 py-1.5 text-sm text-slate-600 hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            <ol className="space-y-6" data-testid="messages">
              {messages.map((m) =>
                m.role === "user" ? (
                  <li key={m.key} className="flex justify-end">
                    <p className="max-w-[85%] rounded-2xl rounded-br-md bg-slate-100 px-4 py-2.5 whitespace-pre-wrap text-slate-800">
                      {m.content}
                    </p>
                  </li>
                ) : (
                  <li key={m.key} data-testid="assistant-message">
                    <AssistantMessage message={m} onOpenCitation={setOpenCitation} />
                  </li>
                ),
              )}
            </ol>
          </div>
        </div>

        <Composer onSend={send} streaming={streaming} onStop={() => abortRef.current?.abort()} />
      </div>

      <CitationPanel citation={openCitation} onClose={closePanel} />
    </div>
  );
}

function AssistantMessage({
  message: m,
  onOpenCitation,
}: {
  message: UiMessage;
  onOpenCitation: (c: Citation) => void;
}) {
  return (
    <div className="text-slate-800">
      {m.streaming && !m.content ? (
        <p className="flex items-center gap-2 text-sm text-slate-400" aria-live="polite">
          <span className="flex gap-1">
            <span className="size-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
            <span className="size-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
            <span className="size-1.5 animate-bounce rounded-full bg-slate-400" />
          </span>
          Reading your documents…
        </p>
      ) : (
        <Markdown content={m.content} citations={m.citations} onOpenCitation={onOpenCitation} />
      )}

      {m.error && (
        <p className="mt-2 flex items-start gap-2 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
          <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />
          {m.error}
        </p>
      )}

      {!!m.citations?.length && (
        <div className="mt-3 flex flex-wrap gap-1.5" aria-label="Sources">
          {m.citations.map((c) => (
            <button
              key={c.number}
              type="button"
              onClick={() => onOpenCitation(c)}
              className="inline-flex max-w-full items-center gap-1.5 rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700"
            >
              <span className="font-semibold text-indigo-600">{c.number}</span>
              <FileText className="size-3.5 shrink-0 text-slate-400" aria-hidden />
              <span className="truncate">{c.filename}</span>
              <span className="shrink-0 text-slate-400">p. {c.page}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Composer({
  onSend,
  streaming,
  onStop,
}: {
  onSend: (q: string) => void;
  streaming: boolean;
  onStop: () => void;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with content up to a limit.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    if (!value.trim() || streaming) return;
    onSend(value);
    setValue("");
  };

  return (
    <div className="border-t border-slate-200 bg-white px-4 py-3 sm:px-6">
      <form
        className="mx-auto flex w-full max-w-3xl items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm focus-within:border-indigo-400 focus-within:ring-2 focus-within:ring-indigo-100"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Ask a question about your documents…"
          aria-label="Question"
          data-testid="question-input"
          className="max-h-[200px] flex-1 resize-none bg-transparent px-2 py-1.5 text-slate-900 outline-none placeholder:text-slate-400"
        />
        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop generating"
            className="rounded-xl bg-slate-800 p-2 text-white hover:bg-slate-700"
          >
            <Square className="size-4" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!value.trim()}
            aria-label="Send"
            data-testid="send-button"
            className="rounded-xl bg-indigo-600 p-2 text-white hover:bg-indigo-700 disabled:bg-slate-200 disabled:text-slate-400"
          >
            <ArrowUp className="size-4" />
          </button>
        )}
      </form>
      <p className="mx-auto mt-1.5 max-w-3xl text-center text-xs text-slate-400">
        Answers use only your documents. Check the cited passages.
      </p>
    </div>
  );
}
