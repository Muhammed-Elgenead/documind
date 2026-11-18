# Production image for the DocuMind API (also used by docker compose).
#
#   docker build -t documind-api backend/
#   docker run -p 8000:8000 -e DATABASE_URL=... -e JWT_SECRET=... -e GEMINI_API_KEY=... documind-api
#
# Hosts that set $PORT (Railway, Render) are supported. Migrations run on start, which
# assumes a single instance (see docs/decisions.md).

# ---------- Build: dependencies, app, and the local models ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# Bake the embedding model and re-ranker into the image (~300 MB): no download on
# start-up, and restarts on hosts with ephemeral disks stay fast.
ARG EMBEDDING_MODEL=ibm-granite/granite-embedding-small-english-r2
ARG RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-12-v2
RUN /opt/venv/bin/python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('${EMBEDDING_MODEL}', cache_dir='/opt/models'); \
TextCrossEncoder('${RERANKER_MODEL}', cache_dir='/opt/models')"

# ---------- Runtime: slim, non-root ----------
# Same base as the build image, so the venv's interpreter path matches.
FROM python:3.12-slim-bookworm AS runtime

RUN useradd --create-home --uid 1000 app
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=app:app /opt/models /opt/models
COPY --from=build --chown=app:app /app /app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_CACHE_DIR=/opt/models \
    UPLOAD_DIR=/tmp/uploads \
    PORT=8000
WORKDIR /app
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# --proxy-headers: behind the host's proxy, take the client IP from X-Forwarded-For
# (the login rate limit is per IP). One worker: rate limits and background ingestion
# live in this process.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
