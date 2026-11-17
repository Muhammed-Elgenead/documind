"""The eval runner (evals/run_evals.py) and its dataset (evals/qa.jsonl)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.agents.llm import LLMError, Usage
from app.agents.prompts import NOT_FOUND_MESSAGE
from app.services.embeddings import HashEmbedder
from app.services.reranking import TermOverlapReranker
from app.services.retrieval import RetrievalMode
from tests.test_retrieval import seed_chunks

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_evals", ROOT / "evals" / "run_evals.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["run_evals"] = ev
_spec.loader.exec_module(ev)


class ScriptedLLM:
    """Returns replies in order; an Exception in the list is raised instead."""

    model = "fake-model"

    def __init__(self, replies, usage=(10, 5)):
        self.replies = list(replies)
        self.usage = usage
        self.calls = []

    async def stream_answer(self, system, messages, usage=None):
        self.calls.append((system, messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if usage is not None:
            usage.input_tokens, usage.output_tokens = self.usage
        yield reply


# ---------- The dataset ----------


def test_qa_dataset_is_well_formed():
    rows = [json.loads(line) for line in (ROOT / "evals" / "qa.jsonl").open(encoding="utf-8")]
    assert len(rows) == 40
    assert len({r["id"] for r in rows}) == 40
    unanswerable = [r for r in rows if not r["sources"]]
    assert len(unanswerable) == 4
    assert all(r["expected_answer"] == NOT_FOUND_MESSAGE for r in unanswerable)
    for r in rows:
        assert r["question"].strip() and r["expected_answer"].strip()
        for s in r["sources"]:
            assert s["document"] in ev.PDFS
            assert isinstance(s["page"], int) and s["page"] >= 1
        if r["sources"]:
            assert r["evidence"], r["id"]
    assert {s["document"] for r in rows for s in r["sources"]} == set(ev.PDFS)


@pytest.mark.skipif(
    not all((ev.DATA / n).exists() for n in ev.PDFS), reason="sample PDFs not downloaded"
)
def test_every_evidence_quote_is_on_its_labelled_page():
    ev.check_pdfs()
    assert len(ev.load_qa()) == 40  # exits with a message on any mismatch


# ---------- LLM calls ----------


async def test_complete_retries_transient_errors(monkeypatch):
    monkeypatch.setattr(ev, "RETRY_DELAYS", (0, 0))
    llm = ScriptedLLM([LLMError("The LLM request failed."), "ok"])
    text, usage = await ev.complete(llm, "sys", [{"role": "user", "content": "q"}])
    assert text == "ok"
    assert usage == Usage(input_tokens=10, output_tokens=5)
    assert len(llm.calls) == 2


async def test_complete_stops_the_run_when_provider_stays_unavailable(monkeypatch):
    monkeypatch.setattr(ev, "RETRY_DELAYS", (0,))
    quota = LLMError("The LLM is rate limited. Please try again shortly.")
    with pytest.raises(ev.ProviderUnavailable, match="rate limited"):
        await ev.complete(ScriptedLLM([quota, quota]), "s", [])


@pytest.mark.parametrize(
    "message", ["The LLM API key is missing or invalid.", "The model declined to answer."]
)
async def test_complete_does_not_retry_permanent_errors(monkeypatch, message):
    monkeypatch.setattr(ev, "RETRY_DELAYS", (0,))
    llm = ScriptedLLM([LLMError(message), "ok"])
    with pytest.raises(LLMError, match=message[:10]):
        await ev.complete(llm, "s", [])
    assert len(llm.calls) == 1


QA = ev.QA(
    id="t-1",
    question="What is the rate?",
    expected_answer="Five percent.",
    sources=[{"document": "a.pdf", "page": 2}],
    evidence="rate",
)


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ('{"score": 4, "reason": "Minor omission."}', 4),
        ('```json\n{"score": 5, "reason": "Exact."}\n```', 5),
        ('{"score": 9, "reason": "?"}', 1),
        ("I think it's fine", 1),
    ],
)
async def test_judge_parses_scores_and_rejects_bad_replies(reply, score):
    llm = ScriptedLLM([reply])
    got, reason, usage = await ev.judge(llm, QA, "It is 5% [1].")
    assert got == score
    assert reason
    system, messages = llm.calls[0]
    assert system == ev.JUDGE_SYSTEM
    assert "Reference answer: Five percent." in messages[0]["content"]
    assert "Assistant's answer: It is 5% [1]." in messages[0]["content"]


def test_cost_and_price_override(monkeypatch):
    assert ev.price_for("gemini-3.8-flash") == (0.75, 3.75)
    assert ev.price_for("unknown-model") is None
    assert ev.cost(Usage(input_tokens=2_000_000, output_tokens=1_000_000), (0.5, 2.0)) == 3.0
    assert ev.cost(Usage(), None) is None
    monkeypatch.setenv("EVAL_PRICE_INPUT_PER_MTOK", "1")
    monkeypatch.setenv("EVAL_PRICE_OUTPUT_PER_MTOK", "4")
    assert ev.price_for("unknown-model") == (1.0, 4.0)


# ---------- Summary and output ----------


def row(**kw):
    base = dict(
        mode="hybrid",
        id="x",
        answerable=True,
        hit_at_5=True,
        cited_expected=True,
        said_not_found=False,
        score=5,
        reason="",
        retrieval_s=0.02,
        total_s=2.0,
        input_tokens=100,
        output_tokens=10,
        cost_usd=0.001,
    )
    return ev.Row(**{**base, **kw})


def test_summarize_separates_answerable_and_unanswerable():
    rows = [
        row(id="a", hit_at_5=True, cited_expected=True, score=5),
        row(id="b", hit_at_5=False, cited_expected=False, score=2, total_s=4.0, cost_usd=0.003),
        row(
            id="c",
            answerable=False,
            hit_at_5=None,
            cited_expected=None,
            said_not_found=True,
            score=5,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        ),
    ]
    s = ev.summarize(rows)
    assert s["hit_at_5"] == 50
    assert s["citation"] == 50
    assert s["not_found"] == 100
    assert s["score"] == pytest.approx(4.0)
    assert s["score_answerable"] == pytest.approx(3.5)
    assert s["total_s"] == pytest.approx(8 / 3)
    assert s["cost"] == pytest.approx(0.004 / 3)
    assert s["llm_skipped"] == 1
    # Unknown price for any row -> no cost figure rather than a misleading average.
    assert ev.summarize([row(cost_usd=None)])["cost"] is None


def test_write_results_writes_files_and_updates_readme(tmp_path, monkeypatch):
    (tmp_path / "evals").mkdir()
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- evals:start -->\nold\n<!-- evals:end -->\nrest\n")
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "EVALS", tmp_path / "evals")

    rows = [row(id="t-1", mode=m) for m in ("vector", "hybrid")]
    summaries = {m: ev.summarize([r for r in rows if r.mode == m]) for m in ("vector", "hybrid")}
    meta = {"date": "2026-11-13", "model": "m", "judge": "m", "price": "$1 / $2", "n_modes": 2}

    # A partial run writes its report but leaves the README alone.
    ev.write_results(rows[:1], summaries, [QA], meta, update_readme=True, partial=True)
    results = (tmp_path / "evals" / "results.md").read_text(encoding="utf-8")
    assert "**Partial run:** 1 of 2 question runs" in results
    assert "old" in readme.read_text(encoding="utf-8")

    ev.write_results(rows, summaries, [QA], meta, update_readme=True)

    results = (tmp_path / "evals" / "results.md").read_text(encoding="utf-8")
    assert "Partial run" not in results
    assert "| `vector` | 100% | 5.00 | 100% |" in results
    assert "| t-1: What is the rate? | ✓ · ✓ · 5 | ✓ · ✓ · 5 |" in results
    lines = (tmp_path / "evals" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text
    assert text.startswith("intro\n<!-- evals:start -->") and text.endswith("evals:end -->\nrest\n")
    assert "| `hybrid` | 100% |" in text


# ---------- One question end to end (real database) ----------


async def test_run_question_scores_retrieval_citations_and_gate(session_factory, test_settings):
    doc_id = await seed_chunks(
        session_factory,
        [(1, "Office plants need watering", None), (2, "The interest rate is five percent", None)],
        filename="a.pdf",
    )
    common = dict(
        session_factory=session_factory,
        embedder=HashEmbedder(),
        reranker=TermOverlapReranker(),
        doc_ids=[doc_id],
        settings=test_settings,
        price=(1.0, 2.0),
    )
    llm = ScriptedLLM(["It is five percent [1]."], usage=(1000, 100))
    judge_llm = ScriptedLLM(['{"score": 5, "reason": "Correct."}'])

    r = await ev.run_question(
        QA, RetrievalMode.HYBRID_RERANK, llm=llm, judge_llm=judge_llm, **common
    )
    assert (r.hit_at_5, r.cited_expected, r.score, r.said_not_found) == (True, True, 5, False)
    assert r.top5[0] == "a.pdf p2"
    assert r.cost_usd == pytest.approx((1000 * 1 + 100 * 2) / 1e6)
    assert r.total_s >= r.retrieval_s > 0

    # With the relevance gate closed: no answer call, the not-found reply is judged.
    gated = test_settings.model_copy(update={"rerank_min_score": 0.99})
    judge_llm = ScriptedLLM(['{"score": 1, "reason": "Declined."}'])
    no_call = ScriptedLLM([])
    r = await ev.run_question(
        QA,
        RetrievalMode.HYBRID_RERANK,
        llm=no_call,
        judge_llm=judge_llm,
        **{**common, "settings": gated},
    )
    assert no_call.calls == []
    assert r.said_not_found and r.answer == NOT_FOUND_MESSAGE
    assert (r.hit_at_5, r.cited_expected, r.input_tokens, r.cost_usd) == (False, False, 0, 0.0)


# ---------- main(): stop on quota, then resume (real database) ----------


async def test_main_stops_cleanly_and_resumes(
    session_factory, test_settings, tmp_path, monkeypatch
):
    doc_id = await seed_chunks(
        session_factory, [(2, "The interest rate is five percent", None)], filename="a.pdf"
    )
    (tmp_path / "evals").mkdir()
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "EVALS", tmp_path / "evals")
    monkeypatch.setattr(ev, "RETRY_DELAYS", ())
    monkeypatch.setattr(ev, "check_pdfs", lambda: None)
    q2 = ev.QA(**{**QA.__dict__, "id": "t-2"})
    monkeypatch.setattr(ev, "load_qa", lambda: [QA, q2])
    monkeypatch.setattr(ev, "Settings", lambda: test_settings)
    monkeypatch.setattr(ev, "create_engine_and_factory", lambda url: (None, session_factory))

    async def fake_ingest(*_):
        return [doc_id]

    monkeypatch.setattr(ev, "ensure_ingested", fake_ingest)
    quota = LLMError("The LLM is rate limited. Please try again shortly.")
    # Answer + judge for t-1, then the quota runs out on t-2's answer.
    llm = ScriptedLLM(["Five percent [1].", '{"score": 5, "reason": "ok"}', quota])
    monkeypatch.setattr(ev, "create_llm", lambda s: llm)
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--modes", "hybrid"])

    await ev.main()
    results = tmp_path / "evals" / "results.jsonl"
    assert [json.loads(line)["id"] for line in results.read_text().splitlines()] == ["t-1"]
    assert "Partial run" in (tmp_path / "evals" / "results.md").read_text(encoding="utf-8")

    # Resume: only t-2 runs; t-1 is kept, not re-asked.
    llm.replies = ["Five percent [1].", '{"score": 4, "reason": "ok"}']
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--modes", "hybrid", "--resume"])
    await ev.main()
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert [(r["id"], r["score"]) for r in rows] == [("t-1", 5), ("t-2", 4)]
    assert "Partial run" not in (tmp_path / "evals" / "results.md").read_text(encoding="utf-8")
