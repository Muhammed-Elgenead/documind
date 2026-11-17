import { CheckCircle2, Loader2, XCircle } from "lucide-react";

import type { DocumentStatus } from "../lib/api";

const styles: Record<DocumentStatus, string> = {
  processing: "bg-amber-50 text-amber-700 ring-amber-200",
  ready: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  failed: "bg-red-50 text-red-700 ring-red-200",
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  const Icon = status === "processing" ? Loader2 : status === "ready" ? CheckCircle2 : XCircle;
  return (
    <span
      data-testid="doc-status"
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ${styles[status]}`}
    >
      <Icon className={`size-3.5 ${status === "processing" ? "animate-spin" : ""}`} aria-hidden />
      {status}
    </span>
  );
}
