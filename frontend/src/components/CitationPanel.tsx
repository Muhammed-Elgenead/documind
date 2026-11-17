import { FileText, X } from "lucide-react";
import { useEffect, useRef } from "react";

import type { Citation } from "../lib/api";
import { reflow } from "../lib/format";

interface Props {
  citation: Citation | null;
  onClose: () => void;
}

/** Side panel (bottom sheet on small screens) showing the exact cited passage. */
export function CitationPanel({ citation, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!citation) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [citation, onClose]);

  if (!citation) return null;

  return (
    <>
      {/* Backdrop on small screens only */}
      <div className="fixed inset-0 z-30 bg-slate-900/30 lg:hidden" onClick={onClose} />
      <aside
        role="complementary"
        aria-label="Cited source"
        data-testid="citation-panel"
        className="fixed inset-x-0 bottom-0 z-40 flex max-h-[75dvh] flex-col rounded-t-2xl border-t border-slate-200 bg-white shadow-2xl lg:static lg:z-auto lg:max-h-none lg:w-96 lg:shrink-0 lg:rounded-none lg:border-t-0 lg:border-l lg:shadow-none"
      >
        <header className="flex items-start gap-3 border-b border-slate-100 px-5 py-4">
          <span className="mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded-md bg-indigo-600 px-1.5 text-xs font-semibold text-white">
            {citation.number}
          </span>
          <div className="min-w-0 flex-1">
            <p className="flex items-center gap-1.5 truncate font-medium text-slate-900">
              <FileText className="size-4 shrink-0 text-slate-400" aria-hidden />
              <span className="truncate" data-testid="citation-filename">
                {citation.filename}
              </span>
            </p>
            <p className="mt-0.5 text-sm text-slate-500" data-testid="citation-page">
              Page {citation.page}
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close source"
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            <X className="size-5" />
          </button>
        </header>
        <div className="overflow-y-auto px-5 py-4">
          <p className="mb-2 text-xs font-medium tracking-wide text-slate-400 uppercase">
            Cited passage
          </p>
          <blockquote
            data-testid="citation-text"
            className="border-l-4 border-indigo-200 bg-indigo-50/40 py-3 pr-3 pl-4 text-sm leading-relaxed whitespace-pre-wrap text-slate-700"
          >
            {reflow(citation.text)}
          </blockquote>
        </div>
      </aside>
    </>
  );
}
