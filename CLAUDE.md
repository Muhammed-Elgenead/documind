# DocuMind

A RAG assistant: users upload PDF/DOCX/MD files, ask questions, and get streamed answers where every claim cites the exact document and page.

## Stack

- **Backend:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2 (async) + Alembic
- **Database:** PostgreSQL 16 with pgvector
- **Frontend:** React + Vite + TypeScript + Tailwind + TanStack Query
- **Infra:** Docker Compose
- **LLM:** Google Gemini (`google-genai` SDK, default `gemini-3.8-flash`) behind the `LLM` interface in `app/agents/llm.py`; the Anthropic SDK remains as an alternative provider. Provider, model and keys come from env vars `LLM_PROVIDER`, `LLM_MODEL`, `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`. Never hardcode them.

## Rules

- Use **uv** for all Python work (`uv init`, `uv add`, `uv run`). Never use pip.
- **Never write anything to the C: drive.** Everything stays inside this project folder on E:, including caches, virtualenvs, temp files and tool output.
- Write **pytest** tests for every endpoint and service. Lint with **ruff**.
- Secrets go only in `.env`. Keep `.env.example` updated with every variable, using placeholder values.
- Record important design decisions in `docs/decisions.md`.
- **Work one phase at a time** (see `PLAN.md`). Stop when the phase's "done when" is met. Never start the next phase on your own.

## Layout

```
backend/
  app/
    api/        # FastAPI routers
    core/       # config, settings, logging
    db/         # models, session, Alembic migrations
    services/   # ingestion, retrieval, embeddings, etc.
    agents/     # LLM orchestration and prompting
  tests/
frontend/
  src/
evals/
docs/
```
