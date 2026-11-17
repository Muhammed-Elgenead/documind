// Turn "[2]" / "[1, 3]" markers in answer markdown into links the renderer shows as chips.

export const CITE_PREFIX = "#cite-";

// Not followed by "(" or ":" so real markdown links / reference definitions are left alone.
const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\](?![(:])/g;

export function linkifyCitations(markdown: string): string {
  // Leave fenced code blocks untouched (odd segments after splitting on ```).
  return markdown
    .split("```")
    .map((segment, i) =>
      i % 2 === 1
        ? segment
        : segment.replace(MARKER, (_, nums: string) =>
            nums
              .split(",")
              .map((n) => `[${n.trim()}](${CITE_PREFIX}${n.trim()})`)
              .join(""),
          ),
    )
    .join("```");
}

export function citationNumber(href: string | undefined): number | null {
  if (!href?.startsWith(CITE_PREFIX)) return null;
  const n = Number(href.slice(CITE_PREFIX.length));
  return Number.isInteger(n) && n > 0 ? n : null;
}
