"""Turn [n] markers in an answer into citation records."""

import re
from collections.abc import Sequence

from app.services.retrieval import RetrievedChunk

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(answer: str, source_count: int) -> list[int]:
    """Unique source numbers cited in the answer, in order of first use.

    Numbers outside 1..source_count (hallucinated sources) are dropped.
    """
    seen: list[int] = []
    for match in _MARKER.finditer(answer):
        for part in match.group(1).split(","):
            n = int(part)
            if 1 <= n <= source_count and n not in seen:
                seen.append(n)
    return seen


def build_citations(answer: str, chunks: Sequence[RetrievedChunk]) -> list[dict]:
    """Citation for each source number actually used. JSON-serializable."""
    return [
        {
            "number": n,
            "document_id": str(chunks[n - 1].document_id),
            "filename": chunks[n - 1].filename,
            "page": chunks[n - 1].page,
            "chunk_id": chunks[n - 1].chunk_id,
            "text": chunks[n - 1].text,
        }
        for n in cited_numbers(answer, len(chunks))
    ]
