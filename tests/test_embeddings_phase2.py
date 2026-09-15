import os

import numpy as np
import pytest

from src.retrieval.embeddings import SentenceTransformerEmbedding


RUN_PHASE2_EMBEDDING_TESTS = (
    os.getenv("RUN_PHASE2_EMBEDDING_TESTS") == "1"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason="Set RUN_PHASE2_EMBEDDING_TESTS=1 to run real BGE-M3 tests",
)
def test_bge_m3_embedding():
    embedder = SentenceTransformerEmbedding(
        model_name="BAAI/bge-m3",
        device="cpu",
    )

    vector = embedder.embed_text(
        "India provides incentives for semiconductor manufacturing."
    )

    assert vector
    assert len(vector) == embedder.dimension
    assert all(np.isfinite(value) for value in vector)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason="Set RUN_PHASE2_EMBEDDING_TESTS=1 to run real BGE-M3 tests",
)
def test_bge_m3_batch_embeddings():
    embedder = SentenceTransformerEmbedding(
        model_name="BAAI/bge-m3",
        device="cpu",
    )

    texts = [
        "India semiconductor manufacturing policy.",
        "Electric vehicle incentives in India.",
        "Global semiconductor supply chains.",
    ]

    vectors = embedder.embed_texts(texts)

    assert len(vectors) == len(texts)
    assert all(len(vector) == embedder.dimension for vector in vectors)
    assert all(
        np.isfinite(value)
        for vector in vectors
        for value in vector
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason="Set RUN_PHASE2_EMBEDDING_TESTS=1 to run real BGE-M3 tests",
)
def test_bge_m3_embeddings_are_normalized():
    embedder = SentenceTransformerEmbedding(
        model_name="BAAI/bge-m3",
        device="cpu",
        normalize_embeddings=True,
    )

    vector = np.asarray(
        embedder.embed_text(
            "Semiconductor manufacturing incentives."
        ),
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    assert np.isclose(norm, 1.0, atol=1e-4)