# DocuMind

**Ask questions about your documents and get answers you can check.** Upload PDFs, Word files or Markdown, ask in plain language, and read a streamed answer in which every claim cites the exact document and page. Click a citation to see the passage it came from.

**Live demo:** _add your Vercel URL here after deploying ([docs/deploy.md](docs/deploy.md))_

<!-- Demo GIF: record ~20 s of upload → ask → click a citation (e.g. with ScreenToGif or Kap),
     save it as docs/demo.gif, then replace the line below with:
     ![DocuMind: upload a PDF, ask a question, click a citation](docs/demo.gif) -->
> 🎬 _Demo GIF goes here: upload a PDF → ask a question → click a citation chip → see the source passage._

## The problem

General-purpose chatbots answer confidently but can't show where an answer came from. That's not good enough for contracts, policies, papers or reports, where the reader has to verify every claim. Pasting documents into a chat window also hits context limits and loses page references.

DocuMind answers **only** from your documents, cites the page for every claim, and says _"I couldn't find this in your documents."_ instead of guessing.

## What it does

- **Upload** PDF, DOCX and Markdown (drag and drop). Files are split into chunks that never cross a page, so every citation points to exactly one page. Indexing runs in the background with live status.
- **Ask** in a chat. Answers stream in token by token, with numbered citation chips. Follow-up questions keep the conversation context.
- **Verify:** clicking a chip opens the file name, page and exact passage, as a side panel on desktop or a bottom sheet on phones.
- **Find the right passages.** Retrieval is hybrid: vector similarity plus Postgres full-text search, fused with Reciprocal Rank Fusion, then re-ranked by a local cross-encoder. It catches both paraphrases ("work from home" → "remote work") and exact terms ("PR.AA-06", "region 24").
- **Know when to say no.** A relevance gate answers "not found" without calling the LLM when nothing relevant was retrieved.
- **Accounts:** email and password sign-in; each user sees only their own documents and conversations. `/chat` is rate-limited per user.

## Architecture

```mermaid
flowchart LR
    U([User]) --> FE

    subgraph Vercel
        FE["React SPA<br/>Vite · TanStack Query · Tailwind"]
    end

    subgraph API["FastAPI backend (Railway, Docker)"]
        AUTH["Auth<br/>Argon2 + JWT · rate limits"]
        ING["Ingestion<br/>parse → chunk per page → embed"]
        RET["Retrieval<br/>vector + full-text → RRF → re-rank → relevance gate"]
        GEN["Answering<br/>grounded prompt → stream → parse citations"]
        MODELS[["Local ONNX models<br/>granite-embedding-small (384-d)<br/>ms-marco-MiniLM cross-encoder"]]
    end

    subgraph Neon["Neon Postgres"]
        DB[("users · documents · chunks<br/>pgvector HNSW + tsvector GIN<br/>conversations · messages")]
    end

    LLM["Google Gemini<br/>(Anthropic Claude optional)"]

    FE -- "HTTPS + Bearer JWT<br/>answers stream as SSE" --> AUTH
    AUTH --> ING & RET
    ING --> MODELS
    RET --> MODELS
    ING --> DB
    RET --> DB
    RET --> GEN
    GEN --> LLM
    GEN -- "answer + citations" --> DB
```

**How a question flows:** the question is embedded locally. The 20 nearest chunks (pgvector, cosine) and the 20 best full-text matches (`ts_rank`) are fused with RRF, and the cross-encoder re-scores the top 20. If the best score is below the threshold, the reply is "not found" with no LLM call. Otherwise the top 8 chunks go to the LLM as numbered sources with strict citation rules. The answer streams back over Server-Sent Events, the `[n]` markers are checked against the sources actually given, and the turn is saved.

## Run it locally

**With Docker** (Postgres, API and frontend with hot reload):

```bash
cp .env.example .env      # set GEMINI_API_KEY (https://aistudio.google.com/apikey) and JWT_SECRET
docker compose up --build
```

Open http://localhost:5173, create an account, upload a file and ask away. The API is at http://localhost:8000 (interactive docs at `/docs`). Migrations run automatically on start.

**Without Docker:** requires [uv](https://docs.astral.sh/uv/) and Node 22+, with Postgres + pgvector running (e.g. `docker compose up db`).

```bash
cd backend && uv sync && uv run alembic upgrade head && uv run uvicorn app.main:app --reload
cd frontend && npm install && npm run dev        # second terminal
```

On Windows, run `source .cache/env.sh` (Git Bash) first to keep tool caches inside the project.

**Production-like, on one machine:** `docker compose -f docker-compose.prod.yml up --build -d` serves the built frontend from nginx at http://localhost:8080.

## Deploy

Neon (Postgres + pgvector) + Railway or Render (API; Render has a free plan) + Vercel (frontend). Step-by-step guide, cost notes and troubleshooting: **[docs/deploy.md](docs/deploy.md)**.

## Evals

There are 40 hand-written questions with reference answers and source pages ([evals/qa.jsonl](evals/qa.jsonl)), over three public PDFs (95 pages): the NIST AI Risk Management Framework, the NIST Cybersecurity Framework 2.0, and "Attention Is All You Need". Four questions have no answer in the documents, to test the "not found" path. Each entry carries a verbatim evidence quote that is checked against its page before every run.

One command compares the three retrieval modes on hit-rate@5, LLM-judged correctness (1–5), citation accuracy, not-found accuracy, latency and cost per query:

```bash
uv run --project backend evals/run_evals.py           # --modes, --limit, --resume
```

The command also writes [evals/results.md](evals/results.md) and fills this table:

<!-- evals:start -->
_No results yet. Run the command above._
<!-- evals:end -->

A full run is ~240 LLM calls (~$1 at Gemini Flash prices), more than the Gemini free tier's ~20 requests per day per model. The runner saves after every question and continues with `--resume`. The sample PDFs aren't committed; the script prints the download commands.

**Retrieval calibration** (Phase 5: 12 labelled questions plus 8 off-topic ones over a handbook and a 50-page report; details in [docs/decisions.md](docs/decisions.md)):

| Mode | Correct page ranked 1st | MRR | Off-topic questions blocked by the gate | Retrieval time* |
|---|---|---|---|---|
| `vector` | 10/12 | 0.868 | 7/8 | ~20 ms |
| `hybrid` | 11/12 | 0.944 | 7/8 | ~20 ms |
| `hybrid_rerank` (default) | **12/12** | **1.000** | **8/8** | ~6 s |

\*On a slow laptop CPU. No relevant question was blocked by the gate in any mode.

## Design decisions

The full reasoning, with measurements, is in [docs/decisions.md](docs/decisions.md). The highlights:

- **Citations you can trust:** chunks never cross a page, so a citation is always one page. The model must cite `[n]` for every claim; numbers that don't match a provided source are dropped, not displayed.
- **Local models instead of APIs** for embeddings and re-ranking (granite-embedding-small, 384-d, 8k context; MiniLM-L-12 cross-encoder). They're free, offline, deterministic and private, and baked into the Docker image.
- **Hybrid retrieval:** vector search alone missed questions that hinge on exact identifiers. Full-text with an OR query and `ts_rank` (not `ts_rank_cd`, which buried rare terms) fixes that; RRF merges the lists without score normalisation.
- **A cross-encoder re-ranker, not an LLM re-ranker:** no extra paid call, no extra network dependency, stable scores, and the clearest separation of relevant from off-topic questions, which is what the relevance gate needs.
- **The LLM is behind a small interface:** Gemini by default, Claude as a config switch, and fakes in tests. SSE events are JSON, and failures arrive as an `error` event after streaming starts.
- **Auth:** Argon2id passwords and stateless JWTs in a bearer header, because the frontend and API are on different domains, where third-party cookies are unreliable. Every query filters by owner, and retrieval requires a `user_id` argument so it can't be forgotten. Another user's data returns 404.
- **Measured before choosing the host:** the two ONNX models peak at ~0.76 GB with batch size 4 (1.6 GB at the old default of 16), so batch sizes are configurable and the guide targets a host without a 512 MB cap.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | – | Backend + database status |
| POST | `/auth/register` | – | `{email, password}` → `{access_token, user}` |
| POST | `/auth/login` | – | `{email, password}` → `{access_token, user}` |
| GET | `/auth/me` | ✓ | The signed-in user |
| POST | `/documents` | ✓ | Upload a PDF/DOCX/MD (multipart `file`, ≤ 50 MB) → 202, indexed in the background |
| GET | `/documents` | ✓ | Your documents with status, newest first |
| GET | `/documents/{id}` | ✓ | Status, `page_count`, `error`, `chunk_count` |
| POST | `/chat` | ✓ | `{question, conversation_id?, document_ids?}` → SSE stream (rate-limited) |
| GET | `/conversations` | ✓ | Your conversations, newest first |
| GET | `/conversations/{id}` | ✓ | A conversation with its messages and citations |

```
event: start   data: {"conversation_id": "…", "sources": 8}
event: token   data: {"text": "Revenue grew 12% "}
event: done    data: {"answer": "Revenue grew 12% [1].", "citations": [{"number": 1, "filename": "report.pdf", "page": 3, "text": "…"}], …}
```

## Tests

```bash
cd backend
uv run ruff check . && uv run ruff format --check .
TEST_DATABASE_URL=postgresql+asyncpg://documind:documind@localhost:5432/documind_test uv run pytest

cd frontend
npm test && npm run build
node e2e/flow.mjs path/to/file.pdf "A question the file answers"   # real browser, needs both servers
```

- **Backend:** pytest covers every endpoint and service: ingestion, retrieval modes and gates, citations, SSE, both LLM providers (with fakes), auth, per-user isolation, rate limits and the eval runner. Database tests run against real Postgres + pgvector when `TEST_DATABASE_URL` is set; tests never download a model.
- **Browser:** the Playwright script drives sign-up → upload → live status → streamed answer → citation panel → reload → mobile layout → a second account seeing nothing.
- **CI** runs all of this (except the browser script, which needs an LLM key), and also builds both production Docker images and smoke-tests the API image against Postgres.

## What I'd do next

- **Run the full eval** on a paid key and use it to tune `RERANK_MIN_SCORE`, chunk size and `top_k`. Then try Gemini/Claude native citation features against the `[n]` prompt convention.
- **Query rewriting for follow-ups**, since retrieval currently uses only the latest question ("and in Q3?" retrieves poorly).
- **A real job queue** (arq/Celery or Postgres `SKIP LOCKED`) for indexing, and Redis for rate limits. Together with running migrations as a release step, that would allow more than one backend instance.
- **Auth hardening:** short-lived access tokens with refresh tokens (so sign-out revokes them), email verification and password reset.
- **Document management:** delete and re-index documents, upload limits per user, and keep the original files in object storage (S3/R2) to show the cited page as an image.
- **OCR** for scanned PDFs, which currently fail with "no extractable text".
- **Observability:** structured logs, request tracing, and LLM token and cost tracking per user.

## Stack

Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · Alembic · PostgreSQL 16 + pgvector · fastembed (ONNX) · Google Gemini (`google-genai`) / Anthropic · React 19 · Vite · TypeScript · Tailwind · TanStack Query · Playwright · Docker · uv · ruff · pytest · vitest

```
backend/app/{api,core,db,services,agents}   backend/tests
frontend/src/{pages,components,lib}         frontend/e2e
evals/   docs/{decisions,deploy}.md
```
