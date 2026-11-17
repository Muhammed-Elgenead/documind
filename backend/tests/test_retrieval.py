"""Phase 5 retrieval: full-text search, RRF fusion, re-ranking, relevance gates, modes."""

import math
import uuid

import pytest
from pydantic import ValidationError

from app.agents.prompts import NOT_FOUND_MESSAGE
from app.core.config import Settings
from app.db.models import EMBEDDING_DIM, Chunk, Document
from app.services.embeddings import HashEmbedder
from app.services.reranking import (
    CrossEncoderReranker,
    TermOverlapReranker,
    create_reranker,
)
from app.services.retrieval import (
    RetrievalMode,
    RetrievalOptions,
    RetrievedChunk,
    keyword_search,
    retrieve,
    rrf_fuse,
)
from tests.conftest import TEST_USER_ID
from tests.test_chat_api import FakeLLM, parse_sse, use_llm

EMB = HashEmbedder()
ALL_MODES = list(RetrievalMode)


def chunk(cid: int, score: float = 0.0, similarity: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=uuid.UUID(int=1),
        filename="f.pdf",
        page=1,
        chunk_index=cid,
        text=f"chunk {cid}",
        score=score,
        similarity=similarity,
    )


def axis(i: int, weight: float = 1.0) -> list[float]:
    """Unit-ish vector pointing mostly along axis i, so cosine similarity is easy to control."""
    v = [0.0] * EMBEDDING_DIM
    v[0] = 1.0
    v[i] = weight
    return v


class FixedQueryEmbedder(HashEmbedder):
    """Every query embeds to `vec`, so tests choose exactly which chunk is nearest."""

    def __init__(self, vec: list[float]):
        super().__init__()
        self.vec = vec

    def embed_query(self, text: str) -> list[float]:
        return self.vec


async def seed_chunks(
    session_factory, rows, filename="doc.pdf", status="ready", user_id=TEST_USER_ID
):
    """rows: [(page, text, embedding or None)]. None embeds the text with HashEmbedder."""
    async with session_factory() as s:
        doc = Document(filename=filename, status=status, page_count=len(rows), user_id=user_id)
        s.add(doc)
        await s.flush()
        for i, (page, text, emb) in enumerate(rows):
            s.add(
                Chunk(
                    document_id=doc.id,
                    page=page,
                    chunk_index=i,
                    text=text,
                    embedding=emb if emb is not None else EMB.embed_query(text),
                )
            )
        await s.commit()
        return doc.id


# ---------- Reciprocal Rank Fusion (pure) ----------


def test_rrf_scores_are_sums_of_reciprocal_ranks():
    vec = [chunk(1, similarity=0.9), chunk(2, similarity=0.8), chunk(3, similarity=0.7)]
    kw = [chunk(3), chunk(4)]
    fused = rrf_fuse([vec, kw], rrf_k=60)

    scores = {c.chunk_id: c.score for c in fused}
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)  # in both lists -> first
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[4] == pytest.approx(1 / 62)
    assert scores[2] == pytest.approx(1 / 62)
    assert [c.chunk_id for c in fused][0] == 3
    # Ties keep the earlier list's order (2 came from the vector list, before 4).
    assert [c.chunk_id for c in fused] == [3, 1, 2, 4]


def test_rrf_keeps_similarity_even_when_keyword_list_saw_chunk_first():
    fused = rrf_fuse([[chunk(7)], [chunk(7, similarity=0.42)]])
    assert fused[0].similarity == 0.42


def test_rrf_handles_empty_lists():
    assert rrf_fuse([[], []]) == []
    assert [c.chunk_id for c in rrf_fuse([[chunk(1)], []])] == [1]


# ---------- Re-rankers (no model download) ----------


def test_term_overlap_reranker_scores_share_of_query_words():
    rr = TermOverlapReranker()
    scores = rr.score("vacation days policy", ["Vacation: 25 days.", "Expenses", "policy days"])
    assert scores == pytest.approx([2 / 3, 0.0, 2 / 3])
    assert rr.score("a?", ["anything"]) == [0.0]  # no usable words


def test_cross_encoder_maps_logits_to_probabilities_in_order():
    class FakeModel:
        def rerank(self, query, texts, batch_size):
            assert query == "q"
            return [0.0, 4.0, -800.0]  # -800 would overflow a naive exp()

    rr = CrossEncoderReranker("any-model", cache_dir=None)  # type: ignore[arg-type]
    rr._model = FakeModel()
    scores = rr.score("q", ["a", "b", "c"])
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(1 / (1 + math.exp(-4)))
    assert scores[2] == pytest.approx(0.0)
    assert rr.score("q", []) == []


def test_create_reranker_rejects_unknown_provider(tmp_path):
    assert isinstance(create_reranker("overlap", "", tmp_path), TermOverlapReranker)
    assert isinstance(create_reranker("fastembed", "m", tmp_path), CrossEncoderReranker)
    with pytest.raises(ValueError, match="RERANKER_PROVIDER"):
        create_reranker("nope", "", tmp_path)


# ---------- Settings ----------


def test_options_from_settings_and_mode_validation():
    s = Settings(
        retrieval_mode="hybrid",
        retrieval_top_k=5,
        retrieval_candidates=30,
        rrf_k=10,
        retrieval_min_similarity=0.4,
        rerank_min_score=0.2,
    )
    assert RetrievalOptions.from_settings(s) == RetrievalOptions(
        mode=RetrievalMode.HYBRID,
        k=5,
        candidates=30,
        rrf_k=10,
        min_similarity=0.4,
        min_rerank_score=0.2,
    )
    with pytest.raises(ValidationError):
        Settings(retrieval_mode="bm25")


# ---------- Full-text search (real database) ----------


async def test_keyword_search_matches_any_word_with_stemming(session_factory):
    await seed_chunks(
        session_factory,
        [
            (1, "We hired forty engineers", None),
            (2, "Hiring freeze ends in March and hiring resumes", None),
            (3, "Office plants need watering", None),
        ],
    )
    async with session_factory() as s:
        # "hiring" stems to "hire" and matches "hired"; ANY word suffices (OR query).
        hits = await keyword_search(s, "When does hiring of engineers restart?", limit=10)
        none = await keyword_search(s, "the of and", limit=10)  # stop words only
    assert {h.page for h in hits} == {1, 2}
    assert hits[0].score >= hits[1].score > 0
    assert all(h.similarity is None for h in hits)
    assert none == []


async def test_keyword_search_respects_status_filter_and_limit(session_factory):
    ready = await seed_chunks(session_factory, [(1, "budget review", None)], "a.pdf")
    await seed_chunks(session_factory, [(1, "budget budget", None)], "b.pdf")
    await seed_chunks(session_factory, [(1, "budget", None)], "pending.pdf", "processing")
    async with session_factory() as s:
        all_hits = await keyword_search(s, "budget", limit=10)
        one = await keyword_search(s, "budget", limit=1)
        only_a = await keyword_search(s, "budget", limit=10, document_ids=[ready])
    assert {h.filename for h in all_hits} == {"a.pdf", "b.pdf"}
    assert len(one) == 1
    assert [h.filename for h in only_a] == ["a.pdf"]


# ---------- The three modes (real database) ----------

# The query vector points along axis 1. "nearest" sits on axis 1 but lacks the code word;
# "coded" is further away but contains the exact identifier the question asks about.
MODE_ROWS = [
    (1, "General notes about error handling in the platform", axis(1, 3.0)),
    (2, "Error ERR-4411 means the license server is unreachable", axis(1, 1.0)),
    (3, "Lunch menu for the week", axis(2, 3.0)),
]
QUESTION = "What does ERR-4411 mean?"


async def test_vector_mode_ranks_by_similarity_only(session_factory):
    await seed_chunks(session_factory, MODE_ROWS)
    emb = FixedQueryEmbedder(axis(1, 3.0))
    async with session_factory() as s:
        res = await retrieve(
            s, emb, QUESTION, RetrievalOptions(mode=RetrievalMode.VECTOR, k=1), user_id=None
        )
    assert [c.page for c in res] == [1]  # the code word doesn't help in pure vector search
    assert res[0].score == res[0].similarity == pytest.approx(1.0)


async def test_hybrid_mode_promotes_exact_keyword_match(session_factory):
    await seed_chunks(session_factory, MODE_ROWS)
    emb = FixedQueryEmbedder(axis(1, 3.0))
    opts = RetrievalOptions(mode=RetrievalMode.HYBRID, k=3, rrf_k=60)
    async with session_factory() as s:
        res = await retrieve(s, emb, QUESTION, opts, user_id=None)
    # Page 2: vector rank 2 + keyword rank 1 beats page 1: vector rank 1 only.
    assert [c.page for c in res][:2] == [2, 1]
    assert res[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert res[0].similarity is not None


class ScriptedReranker:
    """Scores texts by a fixed table and records what it was asked to score."""

    def __init__(self, table: dict[str, float]):
        self.table = table
        self.seen: list[list[str]] = []

    def score(self, query, texts):
        self.seen.append(list(texts))
        return [self.table.get(t, 0.0) for t in texts]


async def test_hybrid_rerank_mode_orders_by_reranker_scores(session_factory):
    await seed_chunks(session_factory, MODE_ROWS)
    emb = FixedQueryEmbedder(axis(1, 3.0))
    texts = {page: text for page, text, _ in MODE_ROWS}
    # Hybrid order is pages 2, 1, 3; the re-ranker prefers 3, then 1, then 2.
    rr = ScriptedReranker({texts[3]: 0.9, texts[1]: 0.6, texts[2]: 0.1})
    async with session_factory() as s:
        hybrid = await retrieve(
            s, emb, QUESTION, RetrievalOptions(mode=RetrievalMode.HYBRID), user_id=None
        )
        res = await retrieve(
            s,
            emb,
            QUESTION,
            RetrievalOptions(mode=RetrievalMode.HYBRID_RERANK, k=2),
            reranker=rr,
            user_id=None,
        )
    assert [c.page for c in hybrid] == [2, 1, 3]
    assert [c.page for c in res] == [3, 1]  # re-ranked, then cut to k
    assert [c.score for c in res] == [0.9, 0.6]  # scores are the re-ranker's
    assert sorted(rr.seen[0]) == sorted(texts.values())  # the whole fused pool was scored


async def test_hybrid_rerank_scores_only_the_top_candidates(session_factory):
    await seed_chunks(session_factory, [(p, f"policy page {p}", None) for p in range(1, 31)])
    rr = ScriptedReranker({})
    opts = RetrievalOptions(mode=RetrievalMode.HYBRID_RERANK, candidates=20)
    async with session_factory() as s:
        await retrieve(s, EMB, "policy", opts, reranker=rr, user_id=None)
    assert len(rr.seen[0]) == 20


async def test_hybrid_rerank_requires_a_reranker(session_factory):
    await seed_chunks(session_factory, MODE_ROWS)
    async with session_factory() as s:
        with pytest.raises(ValueError, match="reranker"):
            await retrieve(
                s, EMB, QUESTION, RetrievalOptions(mode=RetrievalMode.HYBRID_RERANK), user_id=None
            )


@pytest.mark.parametrize("mode", ALL_MODES)
async def test_every_mode_respects_k_status_and_document_filter(session_factory, mode):
    keep = await seed_chunks(
        session_factory,
        [(p, f"revenue report page {p}", None) for p in range(1, 6)],
        "keep.pdf",
    )
    await seed_chunks(session_factory, [(1, "revenue revenue", None)], "other.pdf")
    await seed_chunks(session_factory, [(1, "revenue", None)], "pending.pdf", "processing")
    opts = RetrievalOptions(mode=mode, k=3)
    rr = TermOverlapReranker()
    async with session_factory() as s:
        res = await retrieve(s, EMB, "revenue report", opts, reranker=rr, user_id=None)
        only = await retrieve(
            s, EMB, "revenue report", opts, document_ids=[keep], reranker=rr, user_id=None
        )
    assert len(res) == 3
    assert "pending.pdf" not in {c.filename for c in res}
    assert {c.filename for c in only} == {"keep.pdf"}
    assert [c.score for c in res] == sorted((c.score for c in res), reverse=True)


@pytest.mark.parametrize("mode", ALL_MODES)
async def test_every_mode_returns_nothing_without_ready_documents(session_factory, mode):
    async with session_factory() as s:
        opts = RetrievalOptions(mode=mode)
        assert (
            await retrieve(s, EMB, "anything", opts, reranker=TermOverlapReranker(), user_id=None)
            == []
        )


# ---------- Relevance gate ----------


@pytest.mark.parametrize("mode", [RetrievalMode.VECTOR, RetrievalMode.HYBRID])
async def test_similarity_gate(session_factory, mode):
    await seed_chunks(session_factory, MODE_ROWS)
    # Query along axis 3: no chunk points there, so the best similarity is low.
    far = FixedQueryEmbedder(axis(3, 3.0))
    near = FixedQueryEmbedder(axis(1, 3.0))
    opts = RetrievalOptions(mode=mode, min_similarity=0.5)
    async with session_factory() as s:
        assert await retrieve(s, far, QUESTION, opts, user_id=None) == []
        assert len(await retrieve(s, near, QUESTION, opts, user_id=None)) == 3


async def test_rerank_gate(session_factory):
    await seed_chunks(session_factory, MODE_ROWS)
    opts = RetrievalOptions(mode=RetrievalMode.HYBRID_RERANK, min_rerank_score=0.3)
    rr = TermOverlapReranker()
    async with session_factory() as s:
        # Best overlap: page 2 has "err" and "4411" of {what, does, err, 4411, mean} = 0.4.
        assert len(await retrieve(s, EMB, QUESTION, opts, reranker=rr, user_id=None)) > 0
        # Shares no words with any chunk: best score 0.0.
        assert (
            await retrieve(s, EMB, "Who won the football cup?", opts, reranker=rr, user_id=None)
            == []
        )


# ---------- POST /chat in every mode ----------

# Same texts, embedded by the HashEmbedder the app uses in tests.
TEXT_ROWS = [(page, text, None) for page, text, _ in MODE_ROWS]


@pytest.mark.parametrize("mode", ALL_MODES)
async def test_chat_works_in_every_mode(db_client, session_factory, test_settings, mode):
    test_settings.retrieval_mode = mode.value
    await seed_chunks(session_factory, TEXT_ROWS, "errors.md")
    llm = use_llm(FakeLLM(["It means the license server is unreachable [1]."]))

    events = parse_sse((await db_client.post("/chat", json={"question": QUESTION})).text)

    assert events[-1][0] == "done"
    assert events[0][1]["sources"] == 3
    assert len(llm.calls) == 1
    assert events[-1][1]["citations"][0]["filename"] == "errors.md"


@pytest.mark.parametrize(
    ("mode", "field", "value"),
    [
        ("vector", "retrieval_min_similarity", 1.01),
        ("hybrid", "retrieval_min_similarity", 1.01),
        ("hybrid_rerank", "rerank_min_score", 0.99),
    ],
)
async def test_chat_gate_answers_not_found_without_llm(
    db_client, session_factory, test_settings, mode, field, value
):
    test_settings.retrieval_mode = mode
    setattr(test_settings, field, value)
    await seed_chunks(session_factory, TEXT_ROWS)
    llm = use_llm(FakeLLM(["should not be used"]))

    events = parse_sse((await db_client.post("/chat", json={"question": "Weather?"})).text)

    assert events[0][1]["sources"] == 0
    assert events[-1][0] == "done"
    assert events[-1][1]["answer"] == NOT_FOUND_MESSAGE
    assert events[-1][1]["citations"] == []
    assert llm.calls == []
