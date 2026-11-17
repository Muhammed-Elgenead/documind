import { UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB } from "../lib/api";

interface Props {
  onFiles: (files: File[]) => void;
}

export function Dropzone({ onFiles }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <div
      role="button"
      tabIndex={0}
      data-testid="dropzone"
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (e.dataTransfer.files.length) onFiles(Array.from(e.dataTransfer.files));
      }}
      className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition focus-visible:outline-2 focus-visible:outline-indigo-500 ${
        dragging
          ? "border-indigo-400 bg-indigo-50"
          : "border-slate-300 bg-white hover:border-indigo-300 hover:bg-slate-50"
      }`}
    >
      <UploadCloud
        className={`size-10 ${dragging ? "text-indigo-500" : "text-slate-400"}`}
        aria-hidden
      />
      <p className="font-medium text-slate-800">
        {dragging ? "Drop to upload" : "Drag files here or click to browse"}
      </p>
      <p className="text-sm text-slate-500">
        PDF, DOCX or Markdown · up to {MAX_UPLOAD_MB} MB each
      </p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPTED_EXTENSIONS.join(",")}
        className="hidden"
        data-testid="file-input"
        onChange={(e) => {
          if (e.target.files?.length) onFiles(Array.from(e.target.files));
          e.target.value = "";
        }}
      />
    </div>
  );
}
