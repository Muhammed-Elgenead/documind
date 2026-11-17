import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { Citation } from "../lib/api";
import { citationNumber, linkifyCitations } from "../lib/citations";
import { CitationChip } from "./CitationChip";

interface Props {
  content: string;
  citations: Citation[] | null;
  onOpenCitation: (c: Citation) => void;
}

/** Answer markdown with [n] markers rendered as clickable citation chips. */
export function Markdown({ content, citations, onOpenCitation }: Props) {
  const byNumber = new Map((citations ?? []).map((c) => [c.number, c]));
  return (
    <div className="prose prose-slate max-w-none prose-p:my-2 prose-li:my-0.5 prose-pre:bg-slate-900 prose-headings:mt-4 prose-headings:mb-2">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            const n = citationNumber(href);
            if (n !== null) {
              return (
                <CitationChip number={n} citation={byNumber.get(n)} onOpen={onOpenCitation} />
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {linkifyCitations(content)}
      </ReactMarkdown>
    </div>
  );
}
