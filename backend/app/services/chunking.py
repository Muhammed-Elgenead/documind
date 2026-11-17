"""Split page text into overlapping, token-sized chunks that never cross a page.

Keeping each chunk on a single page means every chunk has exactly one page number,
which is what makes "document + page" citations exact. Short pages produce one
(small) chunk.

Token counts are estimated at ~4 characters per token (typical for English with
modern LLM tokenizers). That keeps the chunker dependency-free; the embedding
model has an 8k-token context, so small estimation errors never truncate.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.extraction import PageText

CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class TextChunk:
    page: int
    chunk_index: int
    text: str


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def chunk_pages(
    pages: Sequence[PageText], chunk_size: int = 500, overlap: int = 50
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[TextChunk] = []
    for page in pages:
        for text in _split_text(page.text, chunk_size, overlap):
            chunks.append(TextChunk(page=page.page, chunk_index=len(chunks), text=text))
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = _words(text, max_chars=chunk_size * CHARS_PER_TOKEN)
    if not words:
        return []

    # prefix[i] = number of characters in words[:i] (words keep trailing whitespace)
    prefix = [0]
    for w in words:
        prefix.append(prefix[-1] + len(w))

    def tokens(i: int, j: int) -> int:
        # Same as estimate_tokens("".join(words[i:j]).strip()), in O(1).
        last = words[j - 1]
        trailing_ws = len(last) - len(last.rstrip())
        return math.ceil((prefix[j] - prefix[i] - trailing_ws) / CHARS_PER_TOKEN)

    pieces: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = start + 1
        while end < n and tokens(start, end + 1) <= chunk_size:
            end += 1
        pieces.append("".join(words[start:end]).strip())
        if end >= n:
            break
        # Step back from `end` until the tail holds ~`overlap` tokens, but always advance.
        next_start = end
        while next_start - 1 > start and tokens(next_start - 1, end) <= overlap:
            next_start -= 1
        start = next_start
    return pieces


def _words(text: str, max_chars: int) -> list[str]:
    """Split into words that keep their trailing whitespace; hard-split oversize words."""
    out: list[str] = []
    for w in re.findall(r"\S+\s*", text):
        while len(w.rstrip()) > max_chars:
            out.append(w[:max_chars] + " ")
            w = w[max_chars:]
        out.append(w)
    return out
