import type { Citation } from "../lib/api";

interface Props {
  number: number;
  citation: Citation | undefined;
  onOpen: (c: Citation) => void;
}

export function CitationChip({ number, citation, onOpen }: Props) {
  // Citations arrive with the final "done" event; until then chips are inert.
  if (!citation) {
    return (
      <span
        className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-slate-100 px-1 align-[0.1em] text-[0.7rem] font-semibold text-slate-400"
        aria-label={`Source ${number}`}
      >
        {number}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={() => onOpen(citation)}
      title={`${citation.filename} · page ${citation.page}`}
      aria-label={`Source ${number}: ${citation.filename}, page ${citation.page}`}
      data-testid="citation-chip"
      className="mx-0.5 inline-flex h-5 min-w-5 cursor-pointer items-center justify-center rounded-md bg-indigo-50 px-1 align-[0.1em] text-[0.7rem] font-semibold text-indigo-700 ring-1 ring-indigo-200 transition hover:bg-indigo-600 hover:text-white focus-visible:outline-2 focus-visible:outline-indigo-500"
    >
      {number}
    </button>
  );
}
