import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, FileText, Loader2, XCircle } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { Dropzone } from "../components/Dropzone";
import { StatusBadge } from "../components/StatusBadge";
import { listDocuments, uploadDocument, validateFile } from "../lib/api";
import { timeAgo } from "../lib/format";

interface PendingUpload {
  key: string;
  name: string;
  error?: string;
}

export function DocumentsPage() {
  const queryClient = useQueryClient();
  const [uploads, setUploads] = useState<PendingUpload[]>([]);

  const documents = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    // Poll while anything is still processing so statuses update live.
    refetchInterval: (q) => (q.state.data?.some((d) => d.status === "processing") ? 2000 : false),
  });

  const upload = useMutation({ mutationFn: uploadDocument });

  function handleFiles(files: File[]) {
    for (const file of files) {
      const key = `${file.name}-${crypto.randomUUID()}`;
      const invalid = validateFile(file);
      setUploads((u) => [...u, { key, name: file.name, error: invalid ?? undefined }]);
      if (invalid) continue;
      upload.mutate(file, {
        onSuccess: () => {
          setUploads((u) => u.filter((x) => x.key !== key));
          queryClient.invalidateQueries({ queryKey: ["documents"] });
        },
        onError: (err) =>
          setUploads((u) => u.map((x) => (x.key === key ? { ...x, error: err.message } : x))),
      });
    }
  }

  const docs = documents.data ?? [];
  const anyReady = docs.some((d) => d.status === "ready");

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6">
        <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
        <p className="mt-1 text-slate-500">
          Upload files to ask questions about them. Answers cite the exact page.
        </p>

        <div className="mt-6">
          <Dropzone onFiles={handleFiles} />
        </div>

        {uploads.length > 0 && (
          <ul className="mt-4 space-y-2">
            {uploads.map((u) => (
              <li
                key={u.key}
                className={`flex items-center gap-3 rounded-xl border px-4 py-3 text-sm ${
                  u.error ? "border-red-200 bg-red-50" : "border-slate-200"
                }`}
              >
                {u.error ? (
                  <XCircle className="size-4 shrink-0 text-red-500" aria-hidden />
                ) : (
                  <Loader2 className="size-4 shrink-0 animate-spin text-indigo-500" aria-hidden />
                )}
                <span className="min-w-0 flex-1 truncate">{u.name}</span>
                {u.error ? (
                  <>
                    <span className="text-red-700">{u.error}</span>
                    <button
                      type="button"
                      className="text-red-700 underline"
                      onClick={() => setUploads((x) => x.filter((y) => y.key !== u.key))}
                    >
                      Dismiss
                    </button>
                  </>
                ) : (
                  <span className="text-slate-500">Uploading…</span>
                )}
              </li>
            ))}
          </ul>
        )}

        <div className="mt-8 flex items-center justify-between">
          <h2 className="font-medium text-slate-900">Your documents</h2>
          {anyReady && (
            <Link
              to="/"
              className="inline-flex items-center gap-1 text-sm font-medium text-indigo-600 hover:text-indigo-700"
            >
              Ask a question <ArrowRight className="size-4" aria-hidden />
            </Link>
          )}
        </div>

        {documents.isPending ? (
          <p className="mt-4 text-sm text-slate-400">Loading…</p>
        ) : documents.isError ? (
          <p className="mt-4 text-sm text-red-600">Could not load documents.</p>
        ) : docs.length === 0 ? (
          <p className="mt-4 rounded-xl border border-dashed border-slate-200 px-4 py-8 text-center text-sm text-slate-400">
            No documents yet. Upload one above.
          </p>
        ) : (
          <ul className="mt-3 divide-y divide-slate-100 rounded-xl border border-slate-200" data-testid="document-list">
            {docs.map((d) => (
              <li key={d.id} className="flex items-start gap-3 px-4 py-3" data-testid="document-row">
                <FileText className="mt-0.5 size-5 shrink-0 text-slate-400" aria-hidden />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-slate-800">{d.filename}</p>
                  <p className="text-xs text-slate-500">
                    {d.page_count != null && `${d.page_count} page${d.page_count === 1 ? "" : "s"} · `}
                    uploaded {timeAgo(d.created_at)}
                  </p>
                  {d.status === "failed" && d.error && (
                    <p className="mt-1 text-xs text-red-600">{d.error}</p>
                  )}
                </div>
                <StatusBadge status={d.status} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
