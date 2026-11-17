const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

export function timeAgo(iso: string, now = Date.now()): string {
  const seconds = Math.round((new Date(iso).getTime() - now) / 1000);
  const abs = Math.abs(seconds);
  if (abs < 60) return rtf.format(seconds, "second");
  if (abs < 3600) return rtf.format(Math.round(seconds / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(seconds / 3600), "hour");
  return rtf.format(Math.round(seconds / 86400), "day");
}

/**
 * Reflow text extracted from PDFs for reading: single line breaks (layout wrapping)
 * become spaces, paragraph breaks stay, and words hyphenated across lines rejoin.
 * Only whitespace changes; the words are exactly what was cited.
 */
export function reflow(text: string): string {
  return text
    .replace(/\r\n?/g, "\n")
    .split(/\n{2,}/)
    .map((para) =>
      para
        .replace(/(\w)-\n(?=[a-z])/g, "$1-") // "Full-\ntime" -> "Full-time"
        .replace(/[ \t]*\n[ \t]*/g, " ")
        .trim(),
    )
    .filter(Boolean)
    .join("\n\n");
}
