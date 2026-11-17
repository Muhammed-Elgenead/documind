import logging
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def normalize_database_url(url: str) -> str:
    """Accept the URLs hosting providers hand out and make them work with asyncpg.

    Neon, Render, Railway and Heroku give `postgres://` or `postgresql://` URLs, often with
    `?sslmode=require&channel_binding=require`. SQLAlchemy needs the `+asyncpg` driver, and
    asyncpg rejects `sslmode` (it takes `ssl`) and doesn't know `channel_binding`.
    """
    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"
    if not scheme.startswith("postgresql+asyncpg"):
        return url
    params = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "sslmode":
            # asyncpg's `ssl` accepts the same mode names (disable, prefer, require, ...).
            params.append(("ssl", value))
        elif key != "channel_binding":
            params.append((key, value))
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


class Settings(BaseSettings):
    # Repo-root .env when run from backend/; a local .env overrides it. Real env vars win.
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str = "postgresql+asyncpg://documind:documind@localhost:5432/documind"
    cors_origins: list[str] = ["http://localhost:5173"]
    # Also allow origins matching this regex, e.g. Vercel preview deployments:
    # https://documind-[a-z0-9-]+\.vercel\.app
    cors_origin_regex: str = ""

    # Auth: HS256-signed JWTs. Set JWT_SECRET in production (e.g. `openssl rand -hex 32`);
    # if empty, a random secret is generated per process and logins don't survive restarts.
    jwt_secret: str = ""
    jwt_expire_minutes: int = 60 * 24 * 7
    # Rate limits, as "count/period" rules separated by ";" (period: second|minute|hour|day).
    # /chat is limited per user; register and login per client IP. Empty = no limit.
    chat_rate_limit: str = "10/minute;200/day"
    auth_rate_limit: str = "10/minute"

    # LLM provider: "gemini" (Google) or "anthropic" (Claude).
    llm_provider: str = "gemini"
    # Empty = the provider's default model (see app.agents.llm.DEFAULT_MODELS).
    llm_model: str = ""
    llm_max_tokens: int = 16000
    # Reasoning depth, passed as-is. Anthropic: low|medium|high|xhigh|max (effort).
    # Gemini: its thinking_level values. Empty = model default.
    llm_effort: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    # Anthropic only: server-side refusal fallback (re-runs a declined request).
    llm_fallbacks: bool = True

    # Retrieval / chat
    # vector | hybrid (vector + full-text, fused with RRF) | hybrid_rerank (+ cross-encoder)
    retrieval_mode: Literal["vector", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    retrieval_top_k: int = 8
    # Candidates taken from each list (vector, full-text) and re-ranked in hybrid_rerank.
    retrieval_candidates: int = 20
    rrf_k: int = 60
    # Relevance gates: below these, answer "not found" without calling the LLM. 0 = off.
    # Defaults are calibrated for the default embedding and re-ranker models
    # (see docs/decisions.md); re-check them when changing models.
    # Cosine similarity of the nearest chunk (vector and hybrid modes).
    retrieval_min_similarity: float = 0.78
    # Re-ranker relevance probability of the best chunk (hybrid_rerank mode).
    rerank_min_score: float = 0.0001
    # Re-ranker: "fastembed" (local cross-encoder) or "overlap" (offline, tests/dev only)
    reranker_provider: str = "fastembed"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-12-v2"
    chat_history_messages: int = 6

    # Uploads
    upload_dir: Path = Path("data/uploads")
    max_upload_mb: int = 50

    # Chunking (tokens are estimated, ~4 characters per token)
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Embeddings: "fastembed" (local ONNX model) or "hash" (offline, tests/dev only)
    embedding_provider: str = "fastembed"
    embedding_model: str = "ibm-granite/granite-embedding-small-english-r2"
    embedding_cache_dir: Path = Path("data/models")
    # Texts per ONNX batch. Peak memory grows with batch size (measured with ~500-token
    # chunks: embedder 274 MB at 1, 449 MB at 4, 1.1 GB at 16; the re-ranker adds 180-500 MB).
    # Lower them on small instances.
    embedding_batch_size: int = 4
    rerank_batch_size: int = 4

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        return normalize_database_url(v)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if not s.jwt_secret:
        logger.warning("JWT_SECRET is not set; using a random one (logins reset on restart).")
        s.jwt_secret = secrets.token_hex(32)
    return s
