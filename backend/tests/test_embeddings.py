import math
import os
from pathlib import Path

import pytest

from app.services.embeddings import (
    Embedder,
    FastEmbedEmbedder,
    HashEmbedder,
    create_embedder,
)


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embedder_shape_norm_and_determinism():
    e = HashEmbedder()
    assert isinstance(e, Embedder)
    [v1, v2] = e.embed_documents(["revenue grew in Q3", "revenue grew in Q3"])
    assert len(v1) == 384
    assert v1 == v2
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0)


def test_hash_embedder_similar_texts_score_higher():
    e = HashEmbedder()
    q = e.embed_query("quarterly revenue growth")
    near, far = e.embed_documents(["revenue growth this quarterly report", "cats sit on mats"])
    assert _cos(q, near) > _cos(q, far)


def test_hash_embedder_never_returns_zero_vector():
    assert any(HashEmbedder().embed_query("!!!"))


def test_create_embedder_selects_provider(tmp_path):
    assert isinstance(create_embedder("hash", "m", tmp_path), HashEmbedder)
    fe = create_embedder("fastembed", "some/model", tmp_path)
    assert isinstance(fe, FastEmbedEmbedder)
    assert fe.model_name == "some/model"
    with pytest.raises(ValueError):
        create_embedder("nope", "m", tmp_path)


def test_fastembed_empty_input_does_not_load_model(tmp_path):
    assert FastEmbedEmbedder("unused/model", tmp_path).embed_documents([]) == []


@pytest.mark.skipif(
    not os.environ.get("FASTEMBED_MODEL_TESTS"),
    reason="Set FASTEMBED_MODEL_TESTS=1 to run the real model (downloads ~180 MB once)",
)
def test_fastembed_real_model():
    cache = Path(os.environ.get("FASTEMBED_CACHE_PATH", "data/models"))
    e = FastEmbedEmbedder("ibm-granite/granite-embedding-small-english-r2", cache)
    q = e.embed_query("How did revenue change?")
    near, far = e.embed_documents(["Revenue rose 12% on cloud growth.", "The cat sat on the mat."])
    assert len(q) == 384
    assert _cos(q, near) > _cos(q, far)
