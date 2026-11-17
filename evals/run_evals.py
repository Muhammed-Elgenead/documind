"""DocuMind evals: compare the retrieval modes on evals/qa.jsonl.

Run from the repository root:

    uv run --project backend evals/run_evals.py

Options:
    --modes vector,hybrid    only these modes (default: all three)
    --limit 5                only the first N questions (smoke test)
    --no-readme              don't update the results table in README.md
    --resume                 keep the rows already in evals/results.jsonl and run only the rest

Results are saved after every question. If the LLM provider stays unavailable (rate limit,
daily quota, overload) after retries, the run stops cleanly with a partial report; run it
again with --resume to continue. The README table is only updated by a complete run.

For each RETRIEVAL_MODE it measures, per question:
  - retrieval hit-rate@5: an expected source (document + page) is in the top 5 chunks
  - answer correctness: an LLM judge scores the answer 1-5 against the reference, with a reason
  - citation accuracy: the answer cites at least one expected source (document + page)
  - not-found accuracy: unanswerable questions get "I couldn't find this in your documents."
  - latency (retrieval, and retrieval + full answer) and cost per query from real token usage

It uses the app's own code (retrieval, prompt, LLM client, citation parsing) and settings
(.env), so results reflect what users get. Writes evals/results.md, evals/results.jsonl
and the table between the eval markers in README.md.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
EVALS = ROOT / "evals"
DATA = EVALS / "data"
sys.path.insert(0, str(BACKEND))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.agents.citations import build_citations  # noqa: E402
from app.agents.llm import LLM, LLMError, Usage, create_llm  # noqa: E402
from app.agents.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_messages  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.db.models import Document, DocumentStatus  # noqa: E402
from app.services.embeddings import create_embedder  # noqa: E402
from app.services.extraction import extract_pages  # noqa: E402
from app.services.ingestion import ingest_document  # noqa: E402
from app.services.reranking import create_reranker  # noqa: E402
from app.services.retrieval import RetrievalMode, RetrievalOptions, retrieve  # noqa: E402

# The sample PDFs. Pinned by hash: page numbers in qa.jsonl are only valid for these files.
PDFS = {
    "nist-ai-rmf-1.0.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1",
    ),
    "nist-csf-2.0.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
        "3c31f46fee98cac0c4323453e5109291a213b4de7fef8c058af9bf67f717433c",
    ),
    "attention-is-all-you-need.pdf": (
        "https://arxiv.org/pdf/1706.03762v7",
        "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
    ),
}

# USD per 1M tokens (input, output). Source: ai.google.dev/gemini-api/docs/pricing,
# checked 2026-11-13 (introductory rate through 2026-12-31; $1.50 / $7.50 from 2027).
# Thinking tokens bill as output. Override: EVAL_PRICE_INPUT_PER_MTOK / ..._OUTPUT_PER_MTOK.
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
}

HIT_K = 5
RETRY_DELAYS = (10, 30, 60)  # seconds, for rate limits and provider overloads

JUDGE_SYSTEM = """You grade answers produced by a document question-answering assistant.

You get a question, a reference answer written by a human from the source documents, and \
the assistant's answer. Judge only whether the assistant's answer is correct and complete \
compared with the reference. Ignore citation markers like [1] and differences in wording \
or length. Extra details are fine unless they contradict the reference.

Scores:
5 = fully correct and complete
4 = correct, with a minor omission or imprecision
3 = partly correct: some key facts right, others missing or vague
2 = mostly wrong or missing the main point, with a little correct content
1 = wrong, contradicts the reference, or fails to answer

If the reference says the information is not in the documents, the correct behaviour is \
to say so: give 5 if the assistant declines for that reason, and 1 if it answers anyway.
If the reference has an answer but the assistant says it couldn't find it, give 1.

Reply with JSON only: {"score": <1-5>, "reason": "<one sentence>"}"""


@dataclass
class QA:
    id: str
    question: str
    expected_answer: str
    sources: list[dict]
    evidence: str | None

    @property
    def answerable(self) -> bool:
        return bool(self.sources)

    @property
    def expected(self) -> set[tuple[str, int]]:
        return {(s["document"], s["page"]) for s in self.sources}


@dataclass
class Row:
    mode: str
    id: str
    answerable: bool
    hit_at_5: bool | None
    cited_expected: bool | None
    said_not_found: bool
    score: int | None
    reason: str
    retrieval_s: float
    total_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    answer: str = ""
    top5: list[str] = field(default_factory=list)
    error: str | None = None


# ---------- Setup and validation ----------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip()


def check_pdfs() -> None:
    missing = [n for n in PDFS if not (DATA / n).exists()]
    if missing:
        lines = [f"Missing sample PDFs in {DATA}:"]
        for n in missing:
            lines.append(f'  curl -L -o "evals/data/{n}" "{PDFS[n][0]}"')
        sys.exit("\n".join(lines))
    for name, (_, sha) in PDFS.items():
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if actual != sha:
            sys.exit(f"{name} is a different version than qa.jsonl was written for (sha256).")


def load_qa() -> list[QA]:
    """Load qa.jsonl and check every evidence quote really is on its labelled page."""
    items = [QA(**json.loads(line)) for line in (EVALS / "qa.jsonl").open(encoding="utf-8")]
    pages = {name: {p.page: _norm(p.text) for p in extract_pages(DATA / name)} for name in PDFS}
    for qa in items:
        for s in qa.sources:
            if s["document"] not in PDFS:
                sys.exit(f"{qa.id}: unknown document {s['document']!r}")
        if qa.answerable:
            first = qa.sources[0]
            if _norm(qa.evidence or "") not in pages[first["document"]].get(first["page"], ""):
                sys.exit(f"{qa.id}: evidence not found on {first['document']} p{first['page']}")
    return items


async def ensure_ingested(settings: Settings, session_factory, embedder) -> list[uuid.UUID]:
    """Use the newest ready copy of each sample PDF, ingesting any that is missing."""
    ids = []
    for name in PDFS:
        async with session_factory() as s:
            doc = (
                await s.execute(
                    select(Document)
                    .where(Document.filename == name, Document.status == DocumentStatus.READY)
                    .order_by(Document.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if doc is not None:
            ids.append(doc.id)
            continue
        print(f"Ingesting {name} (one-time; embeds every page)...", flush=True)
        doc_id = uuid.uuid4()
        upload_dir = settings.upload_dir
        if not upload_dir.is_absolute():
            upload_dir = BACKEND / upload_dir  # the app resolves it from backend/
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / f"{doc_id}.pdf"
        shutil.copyfile(DATA / name, dest)
        async with session_factory() as s:
            s.add(Document(id=doc_id, filename=name, status=DocumentStatus.PROCESSING))
            await s.commit()
        await ingest_document(
            doc_id,
            dest,
            session_factory,
            embedder,
            chunk_size=settings.chunk_size_tokens,
            overlap=settings.chunk_overlap_tokens,
        )
        async with session_factory() as s:
            doc = await s.get(Document, doc_id)
        if doc is None or doc.status != DocumentStatus.READY:
            sys.exit(f"Ingesting {name} failed: {doc.error if doc else 'document vanished'}")
        ids.append(doc_id)
    return ids


# ---------- LLM calls ----------

# LLMError messages (see app.agents.llm) that mean "try again later", not "this request is bad".
TRANSIENT = ("rate limited", "request failed", "Could not reach")


class ProviderUnavailable(Exception):
    """The provider kept failing after retries (quota, overload): stop the run, keep results."""


async def complete(llm: LLM, system: str, messages: list[dict]) -> tuple[str, Usage]:
    """Run a full (streamed) completion, retrying transient provider failures."""
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        usage = Usage()
        try:
            parts = [t async for t in llm.stream_answer(system, messages, usage)]
            return "".join(parts).strip(), usage
        except LLMError as exc:
            if not any(t in str(exc) for t in TRANSIENT):
                raise  # e.g. an invalid key or a refusal: retrying won't help
            if delay is None:
                raise ProviderUnavailable(str(exc)) from exc
            print(f"    {exc} Retrying in {delay}s (attempt {attempt + 2})...", flush=True)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def judge(llm: LLM, qa: QA, answer: str) -> tuple[int, str, Usage]:
    prompt = (
        f"Question: {qa.question}\n\nReference answer: {qa.expected_answer}\n\n"
        f"Assistant's answer: {answer}"
    )
    text, usage = await complete(llm, JUDGE_SYSTEM, [{"role": "user", "content": prompt}])
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
        score = int(data["score"])
        if not 1 <= score <= 5:
            raise ValueError(score)
        return score, str(data.get("reason", "")).strip(), usage
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return 1, f"Unparseable judge reply: {text[:200]!r}", usage


def price_for(model: str) -> tuple[float, float] | None:
    env_in = os.environ.get("EVAL_PRICE_INPUT_PER_MTOK")
    env_out = os.environ.get("EVAL_PRICE_OUTPUT_PER_MTOK")
    if env_in and env_out:
        return float(env_in), float(env_out)
    return PRICES.get(model)


def cost(usage: Usage, price: tuple[float, float] | None) -> float | None:
    if price is None:
        return None
    return (usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1_000_000


# ---------- The run ----------


async def run_question(
    qa, mode, session_factory, embedder, reranker, doc_ids, settings, llm, judge_llm, price
) -> Row:
    opts = replace(RetrievalOptions.from_settings(settings), mode=mode)
    t0 = time.perf_counter()
    async with session_factory() as s:
        chunks = await retrieve(
            s,
            embedder,
            qa.question,
            opts,
            document_ids=doc_ids,
            reranker=reranker,
            user_id=None,  # the eval PDFs have no owner
        )
    t_retrieval = time.perf_counter() - t0

    usage = Usage()
    error = None
    if not chunks:
        answer = NOT_FOUND_MESSAGE  # the relevance gate: no LLM call, as in /chat
    else:
        try:
            answer, usage = await complete(
                llm, SYSTEM_PROMPT, build_messages([], qa.question, chunks)
            )
        except LLMError as exc:
            answer, error = "", str(exc)
    t_total = time.perf_counter() - t0

    citations = build_citations(answer, chunks)
    cited = {(c["filename"], c["page"]) for c in citations}
    top5 = [(c.filename, c.page) for c in chunks[:HIT_K]]
    said_not_found = NOT_FOUND_MESSAGE.rstrip(".").lower() in answer.lower()

    if error:  # e.g. the model declined: a real failure of this question, scored as such
        score, reason, judge_usage = 1, f"Answer failed: {error}", Usage()
    else:
        score, reason, judge_usage = await judge(judge_llm, qa, answer)

    return Row(
        mode=mode.value,
        id=qa.id,
        answerable=qa.answerable,
        hit_at_5=any(t in qa.expected for t in top5) if qa.answerable else None,
        cited_expected=bool(cited & qa.expected) if qa.answerable else None,
        said_not_found=said_not_found,
        score=score,
        reason=reason,
        retrieval_s=t_retrieval,
        total_s=t_total,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost(usage, price),
        judge_input_tokens=judge_usage.input_tokens,
        judge_output_tokens=judge_usage.output_tokens,
        answer=answer,
        top5=[f"{d} p{p}" for d, p in top5],
        error=error,
    )


def summarize(rows: list[Row]) -> dict:
    ans = [r for r in rows if r.answerable]
    unans = [r for r in rows if not r.answerable]
    costs = [r.cost_usd for r in rows if r.cost_usd is not None]

    def pct(xs):
        return 100 * sum(xs) / len(xs) if xs else float("nan")

    return {
        "n": len(rows),
        "hit_at_5": pct([r.hit_at_5 for r in ans]),
        "score": statistics.mean(r.score for r in rows),
        "score_answerable": statistics.mean(r.score for r in ans) if ans else float("nan"),
        "citation": pct([r.cited_expected for r in ans]),
        "not_found": pct([r.said_not_found for r in unans]),
        "retrieval_ms": 1000 * statistics.mean(r.retrieval_s for r in rows),
        "total_s": statistics.mean(r.total_s for r in rows),
        "cost": statistics.mean(costs) if len(costs) == len(rows) else None,
        "llm_skipped": sum(1 for r in rows if r.input_tokens == 0 and not r.error),
        "errors": sum(1 for r in rows if r.error),
    }


def table(summaries: dict[str, dict]) -> str:
    head = (
        "| Mode | Hit@5 | Correctness (1-5) | Citation accuracy | Not-found accuracy "
        "| Avg latency (retrieval / total) | Cost / query |\n"
        "|---|---|---|---|---|---|---|\n"
    )

    def pct(x: float) -> str:
        return "–" if x != x else f"{x:.0f}%"  # NaN: no questions of that kind

    lines = []
    for mode, s in summaries.items():
        cost_text = f"${s['cost']:.5f}" if s["cost"] is not None else "n/a"
        lines.append(
            f"| `{mode}` | {pct(s['hit_at_5'])} | {s['score']:.2f} | {pct(s['citation'])} "
            f"| {pct(s['not_found'])} | {s['retrieval_ms']:.0f} ms / {s['total_s']:.1f} s "
            f"| {cost_text} |"
        )
    return head + "\n".join(lines) + "\n"


def write_results(
    rows: list[Row],
    summaries: dict,
    qa: list[QA],
    meta: dict,
    update_readme: bool,
    partial: bool = False,
) -> None:
    tbl = table(summaries)
    n_ans = sum(q.answerable for q in qa)
    modes = list(summaries)
    judge_in = sum(r.judge_input_tokens for r in rows)
    judge_out = sum(r.judge_output_tokens for r in rows)

    md = [
        "# Eval results",
        "",
        f"Run {meta['date']} · answer model `{meta['model']}` · judge `{meta['judge']}` · "
        f"{len(qa)} questions ({n_ans} answerable, {len(qa) - n_ans} not in the documents) "
        "over 3 PDFs (95 pages). Generated by `uv run --project backend evals/run_evals.py`.",
        "",
    ]
    if partial:
        md += [
            f"> **Partial run:** {len(rows)} of {len(qa) * meta['n_modes']} question runs "
            "finished before the LLM provider became unavailable. Continue with `--resume`.",
            "",
        ]
    md += [
        tbl,
        "**Columns**",
        "",
        f"- **Hit@{HIT_K}:** share of answerable questions with an expected page "
        f"(document + page) among the top {HIT_K} retrieved chunks.",
        "- **Correctness:** mean LLM-judge score, 1-5, of the answer against the reference "
        "answer (all questions).",
        "- **Citation accuracy:** share of answerable questions whose answer cites at least "
        "one expected page.",
        "- **Not-found accuracy:** share of unanswerable questions answered with "
        '"I couldn\'t find this in your documents."',
        "- **Latency:** mean wall time of retrieval, and of retrieval + the full streamed "
        "answer. Measured on the machine that ran the evals.",
        "- **Cost / query:** answer-generation tokens × list price "
        f"({meta['price']}). Retrieval runs locally and costs nothing; the judge is not "
        "included.",
        "",
        "**Run details**",
        "",
    ]
    for mode, s in summaries.items():
        md.append(
            f"- `{mode}`: correctness on answerable questions {s['score_answerable']:.2f}; "
            f"LLM call skipped by the relevance gate {s['llm_skipped']}×; "
            f"failed answers {s['errors']}; questions scored {s['n']}."
        )
    md += [
        f"- Judge usage for the whole run: {judge_in:,} input / {judge_out:,} output tokens.",
        "",
        "## Per question",
        "",
        "Each cell: hit@5 · cited expected page · judge score.",
        "",
        "| Question | " + " | ".join(f"`{m}`" for m in modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    by_key = {(r.id, r.mode): r for r in rows}
    for q in qa:
        cells = []
        for m in modes:
            r = by_key.get((q.id, m))
            if r is None:
                cells.append("")
            elif q.answerable:
                yes = lambda b: "✓" if b else "✗"  # noqa: E731
                cells.append(f"{yes(r.hit_at_5)} · {yes(r.cited_expected)} · {r.score}")
            else:
                cells.append(f"{'not found' if r.said_not_found else 'answered'} · {r.score}")
        md.append(f"| {q.id}: {q.question} | " + " | ".join(cells) + " |")
    md += ["", "Judge reasons and full answers are in `evals/results.jsonl`.", ""]
    (EVALS / "results.md").write_text("\n".join(md), encoding="utf-8")

    with (EVALS / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    if update_readme and not partial:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8") if readme.exists() else ""
        start, end = "<!-- evals:start -->", "<!-- evals:end -->"
        if start in text and end in text:
            block = (
                f"{start}\n_{meta['date']} · `{meta['model']}` · {len(qa)} questions · "
                f"full results in [evals/results.md](evals/results.md)_\n\n{tbl}{end}"
            )
            pattern = re.escape(start) + ".*?" + re.escape(end)
            text = re.sub(pattern, lambda _: block, text, flags=re.DOTALL)
            readme.write_text(text, encoding="utf-8")
        else:
            print("README.md has no eval markers; skipped updating it.")


def create_engine_and_factory(url: str):
    engine = create_async_engine(url)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modes", default=",".join(m.value for m in RetrievalMode))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Windows consoles default to a legacy code page that can't print the progress marks.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    modes = [RetrievalMode(m.strip()) for m in args.modes.split(",")]

    logging.basicConfig(level=logging.WARNING)
    check_pdfs()
    qa = load_qa()[: args.limit]

    settings = Settings()
    if url := os.environ.get("EVAL_DATABASE_URL"):
        settings = settings.model_copy(update={"database_url": url})
    llm = create_llm(settings)
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "")
    judge_settings = settings.model_copy(update={"llm_model": judge_model or settings.llm_model})
    judge_llm = create_llm(judge_settings)
    model = getattr(llm, "model", settings.llm_model)
    price = price_for(model)

    embedder = create_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_cache_dir,
        settings.embedding_batch_size,
    )
    reranker = create_reranker(
        settings.reranker_provider,
        settings.reranker_model,
        settings.embedding_cache_dir,
        settings.rerank_batch_size,
    )
    engine, session_factory = create_engine_and_factory(settings.database_url)
    try:
        doc_ids = await ensure_ingested(settings, session_factory, embedder)
        # Load models now so the first question's latency doesn't include it.
        await asyncio.to_thread(embedder.embed_query, "warm up")
        if RetrievalMode.HYBRID_RERANK in modes:
            await asyncio.to_thread(reranker.score, "warm up", ["warm up"])

        results_path = EVALS / "results.jsonl"
        wanted = {(m.value, q.id) for m in modes for q in qa}
        rows: list[Row] = []
        if args.resume and results_path.exists():
            for line in results_path.open(encoding="utf-8"):
                r = Row(**json.loads(line))
                if (r.mode, r.id) in wanted:
                    rows.append(r)
            print(f"Resuming: {len(rows)} question runs already done.", flush=True)
        done = {(r.mode, r.id) for r in rows}
        with results_path.open("w", encoding="utf-8") as f:  # keep only the rows we resume
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        stopped = False
        for mode in modes:
            if stopped:
                break
            print(f"\n== {mode.value} ({len(qa)} questions)", flush=True)
            for i, q in enumerate(qa, 1):
                if (mode.value, q.id) in done:
                    continue
                try:
                    row = await run_question(
                        q, mode, session_factory, embedder, reranker, doc_ids, settings,
                        llm, judge_llm, price,
                    )  # fmt: skip
                except ProviderUnavailable as exc:
                    print(f"\nStopping: the LLM provider is unavailable ({exc})", flush=True)
                    stopped = True
                    break
                rows.append(row)
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                mark = "✓" if row.hit_at_5 else ("·" if row.hit_at_5 is None else "✗")
                print(
                    f"  [{i:>2}/{len(qa)}] {q.id:<8} hit@5 {mark}  score {row.score}  "
                    f"{row.total_s:5.1f}s",
                    flush=True,
                )
    finally:
        if engine is not None:
            await engine.dispose()

    partial = len(rows) < len(wanted)
    summaries = {
        m.value: summarize(mode_rows)
        for m in modes
        if (mode_rows := [r for r in rows if r.mode == m.value])
    }
    if not summaries:
        sys.exit("No questions finished, so there is nothing to report. Try again later.")
    meta = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "model": model,
        "judge": getattr(judge_llm, "model", judge_model or model),
        "n_modes": len(modes),
        "price": (
            f"${price[0]:g} / ${price[1]:g} per 1M input / output tokens"
            if price
            else "no price known for this model; set EVAL_PRICE_INPUT_PER_MTOK and "
            "EVAL_PRICE_OUTPUT_PER_MTOK"
        ),
    }
    write_results(rows, summaries, qa, meta, update_readme=not args.no_readme, partial=partial)
    print("\n" + table(summaries))
    print(f"Wrote {EVALS / 'results.md'}")
    if partial:
        print("Partial run: README not updated. Continue later with --resume.")


if __name__ == "__main__":
    asyncio.run(main())
