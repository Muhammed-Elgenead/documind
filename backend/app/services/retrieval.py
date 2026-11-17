"""Retrieval over ingested chunks: vector, hybrid (vector + full-text, RRF), and re-ranked.

Modes (RETRIEVAL_MODE):
  vector         top-k by cosine similarity.
  hybrid         vector and full-text candidate lists fused with Reciprocal Rank Fusion.
  hybrid_rerank  hybrid, then the top fused candidates re-scored by a cross-encoder.

Each mode has a relevance gate: if the best candidate scores below a threshold,
nothing is returned and the caller answers "not found" without calling the LLM.
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from sqlalchemy import Text, cast, func, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Chunk, Document, DocumentStatus
from app.services.embeddings import Embedder
from app.services.reranking import Reranker

logger = logging.getLogger(__name__)


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True)
class RetrievalOptions:
    mode: RetrievalMode = RetrievalMode.VECTOR
    k: int = 8
    # Size of each candidate list (vector, full-text) and of the re-ranked pool.
    candidates: int = 20
    # RRF constant: score = sum(1 / (rrf_k + rank)). 60 is the value from the RRF paper.
    rrf_k: int = 60
    # Gate for vector and hybrid: best cosine similarity must reach this.
    min_similarity: float = 0.0
    # Gate for hybrid_rerank: best re-ranker probability must reach this.
    min_rerank_score: float = 0.0

    @classmethod
    def from_settings(cls, s: Settings) -> "RetrievalOptions":
        return cls(
            mode=RetrievalMode(s.retrieval_mode),
            k=s.retrieval_top_k,
            candidates=s.retrieval_candidates,
            rrf_k=s.rrf_k,
            min_similarity=s.retrieval_min_similarity,
            min_rerank_score=s.rerank_min_score,
        )


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    page: int
    chunk_index: int
    text: str
    # Ranking score of the mode that produced it, higher is better:
    # cosine similarity (vector), RRF score (hybrid) or re-ranker probability (hybrid_rerank).
    score: float
    # Cosine similarity to the question, when the chunk was a vector candidate.
    similarity: float | None = None


def _base_query(user_id: uuid.UUID | None, *columns):
    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Document.filename,
            Chunk.page,
            Chunk.chunk_index,
            Chunk.text,
            *columns,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == DocumentStatus.READY)
    )
    if user_id is not None:
        stmt = stmt.where(Document.user_id == user_id)
    return stmt


def _row_to_chunk(r, score: float, similarity: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=r.id,
        document_id=r.document_id,
        filename=r.filename,
        page=r.page,
        chunk_index=r.chunk_index,
        text=r.text,
        score=score,
        similarity=similarity,
    )


async def vector_search(
    session: AsyncSession,
    query_vec: list[float],
    limit: int,
    document_ids: Sequence[uuid.UUID] | None = None,
    user_id: uuid.UUID | None = None,
) -> list[RetrievedChunk]:
    """Nearest chunks by cosine distance (HNSW index)."""
    distance = Chunk.embedding.cosine_distance(query_vec)
    stmt = _base_query(user_id, distance.label("distance")).order_by(distance).limit(limit)
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    rows = (await session.execute(stmt)).all()
    out = []
    for r in rows:
        sim = 1.0 - float(r.distance)
        out.append(_row_to_chunk(r, score=sim, similarity=sim))
    return out


def _any_word_tsquery(question: str):
    """Full-text query matching chunks that contain ANY of the question's words.

    plainto_tsquery ANDs every word, which almost never matches a natural-language
    question. Swapping '&' for '|' keeps its stemming and stop-word removal while
    making it an OR query, ranked with ts_rank (how often the matched words occur).
    """
    anded = cast(func.plainto_tsquery("english", question), Text)
    return cast(func.replace(anded, " & ", " | "), TSQUERY)


async def keyword_search(
    session: AsyncSession,
    question: str,
    limit: int,
    document_ids: Sequence[uuid.UUID] | None = None,
    user_id: uuid.UUID | None = None,
) -> list[RetrievedChunk]:
    """Chunks matching the question's words, by ts_rank (GIN index on chunks.tsv).

    Not ts_rank_cd: with an OR query its cover-density score rewards chunks that repeat
    common words ("region", "figure") over the one chunk with the rare word ("42").
    """
    tsq = _any_word_tsquery(question)
    rank = func.ts_rank(Chunk.tsv, tsq)
    stmt = (
        _base_query(user_id, rank.label("rank"))
        .where(Chunk.tsv.op("@@")(tsq))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    rows = (await session.execute(stmt)).all()
    return [_row_to_chunk(r, score=float(r.rank)) for r in rows]


def rrf_fuse(rankings: Sequence[Sequence[RetrievedChunk]], rrf_k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (rrf_k + rank), rank from 1.

    Only ranks matter, so cosine similarities and ts_rank values (different scales)
    combine without normalisation. A chunk found by both lists keeps its similarity.
    """
    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, c in enumerate(ranking, start=1):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            prev = chunks.get(c.chunk_id)
            if prev is None or (prev.similarity is None and c.similarity is not None):
                chunks[c.chunk_id] = c
    # Ties (same RRF score) keep the earlier list's order: dicts preserve insertion order.
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [replace(chunks[cid], score=scores[cid]) for cid in order]


async def retrieve(
    session: AsyncSession,
    embedder: Embedder,
    question: str,
    options: RetrievalOptions | None = None,
    document_ids: Sequence[uuid.UUID] | None = None,
    reranker: Reranker | None = None,
    *,
    user_id: uuid.UUID | None,
) -> list[RetrievedChunk]:
    """Top-k chunks from ready documents, or [] if nothing passes the relevance gate.

    `user_id` limits the search to that user's documents. It is required so no caller
    can forget it; only internal tools (evals) pass None, meaning any owner.
    """
    opts = options or RetrievalOptions()
    query_vec = await asyncio.to_thread(embedder.embed_query, question)

    if opts.mode == RetrievalMode.VECTOR:
        hits = await vector_search(session, query_vec, opts.k, document_ids, user_id)
        return _gate_on_similarity(hits, hits, opts)

    vec_hits = await vector_search(session, query_vec, opts.candidates, document_ids, user_id)
    kw_hits = await keyword_search(session, question, opts.candidates, document_ids, user_id)
    fused = rrf_fuse([vec_hits, kw_hits], opts.rrf_k)

    if opts.mode == RetrievalMode.HYBRID:
        return _gate_on_similarity(fused[: opts.k], vec_hits, opts)

    if reranker is None:
        raise ValueError("hybrid_rerank mode needs a reranker")
    pool = fused[: opts.candidates]
    if not pool:
        return []
    scores = await asyncio.to_thread(reranker.score, question, [c.text for c in pool])
    reranked = sorted(
        (replace(c, score=s) for c, s in zip(pool, scores, strict=True)),
        key=lambda c: c.score,
        reverse=True,
    )
    best = reranked[0].score
    if best < opts.min_rerank_score:
        logger.info("Retrieval gate: best re-rank score %.2g < %.2g", best, opts.min_rerank_score)
        return []
    return reranked[: opts.k]


def _gate_on_similarity(
    results: list[RetrievedChunk], vec_hits: list[RetrievedChunk], opts: RetrievalOptions
) -> list[RetrievedChunk]:
    """Vector and hybrid gate on the best cosine similarity.

    RRF scores only encode rank, not how relevant anything is, so hybrid uses the
    similarity of the nearest vector candidate as its absolute relevance signal.
    """
    if not vec_hits:
        return []
    best = max(c.similarity or 0.0 for c in vec_hits)
    if best < opts.min_similarity:
        logger.info("Retrieval gate: best similarity %.3f < %.3f", best, opts.min_similarity)
        return []
    return results
