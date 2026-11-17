"""Re-rankers: score (question, passage) pairs jointly, more precisely than embeddings.

Swap providers with RERANKER_PROVIDER. Scores are relevance probabilities in [0, 1],
so one threshold (RERANK_MIN_SCORE) means the same thing whatever the model.
"""

import math
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.config import get_settings


@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query, in [0, 1], same order as `texts`."""
        ...


def _sigmoid(x: float) -> float:
    # Numerically safe for large |x|.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderReranker:
    """Local ONNX cross-encoder via fastembed. No API key; runs on CPU.

    The model outputs raw logits; a sigmoid maps them to probabilities.
    It downloads into `cache_dir` on first use and is reused after.
    """

    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 4):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder  # heavy; lazy

                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextCrossEncoder(self.model_name, cache_dir=str(self.cache_dir))
            return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = self._get_model().rerank(query, texts, batch_size=self.batch_size)
        return [_sigmoid(float(x)) for x in logits]


_WORD = re.compile(r"\w{3,}")


class TermOverlapReranker:
    """Share of the query's words (3+ letters) found in the text. Tests/offline dev only.

    Deterministic and dependency-free, so tests exercise the re-ranking path
    without downloading a model. Not for real relevance.
    """

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(_WORD.findall(query.lower()))
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(_WORD.findall(t.lower()))) / len(q) for t in texts]


def create_reranker(
    provider: str, model_name: str, cache_dir: Path, batch_size: int = 4
) -> Reranker:
    if provider == "fastembed":
        return CrossEncoderReranker(
            model_name=model_name, cache_dir=cache_dir, batch_size=batch_size
        )
    if provider == "overlap":
        return TermOverlapReranker()
    raise ValueError(f"Unknown RERANKER_PROVIDER: {provider!r}")


@lru_cache
def get_reranker() -> Reranker:
    s = get_settings()
    return create_reranker(
        s.reranker_provider, s.reranker_model, s.embedding_cache_dir, s.rerank_batch_size
    )
