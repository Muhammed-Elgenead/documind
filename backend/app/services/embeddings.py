"""Embedding providers behind one small interface.

Swap providers with EMBEDDING_PROVIDER. Anything implementing `Embedder` works,
e.g. a future Voyage AI or OpenAI client. The vector size must match
`app.db.models.EMBEDDING_DIM`; a different size needs a migration.
"""

import hashlib
import math
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.config import get_settings


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models use a different prompt for queries)."""
        ...


class FastEmbedEmbedder:
    """Local ONNX model via fastembed. No API key; runs on CPU.

    The model (~180 MB) downloads into `cache_dir` on first use and is reused after.
    """

    def __init__(self, model_name: str, cache_dir: Path, dim: int = 384, batch_size: int = 4):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.dim = dim
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding  # heavy import; load lazily

                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        return [v.tolist() for v in model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._get_model().query_embed([text]))).tolist()


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests and offline dev only.

    Texts sharing words get similar vectors, which is enough to exercise
    the pipeline without downloading a model. Not for real retrieval quality.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            vec[0] = 1.0  # pgvector cosine distance is undefined for zero vectors
            return vec
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_embedder(
    provider: str, model_name: str, cache_dir: Path, batch_size: int = 4
) -> Embedder:
    if provider == "fastembed":
        return FastEmbedEmbedder(model_name=model_name, cache_dir=cache_dir, batch_size=batch_size)
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return create_embedder(
        s.embedding_provider, s.embedding_model, s.embedding_cache_dir, s.embedding_batch_size
    )
