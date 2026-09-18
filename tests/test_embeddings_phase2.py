import os

import numpy as np
import pytest

from src.retrieval.embeddings import SentenceTransformerEmbedding


RUN_PHASE2_EMBEDDING_TESTS = (
    os.getenv("RUN_PHASE2_EMBEDDING_TESTS") == "1"
)


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedding:
    """Load the cached BGE-M3 snapshot once for the integration tests."""

    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        "BAAI/bge-m3",
        local_files_only=True,
    )

    return SentenceTransformerEmbedding(
        model_name=model_path,
        device="cpu",
        normalize_embeddings=True,
        batch_size=8,
        max_seq_length=1024,
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason=(
        "Set RUN_PHASE2_EMBEDDING_TESTS=1 "
        "to run real BGE-M3 tests"
    ),
)
def test_bge_m3_dimension(
    embedder: SentenceTransformerEmbedding,
) -> None:
    assert embedder.dimension == 1024


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason=(
        "Set RUN_PHASE2_EMBEDDING_TESTS=1 "
        "to run real BGE-M3 tests"
    ),
)
def test_bge_m3_single_embedding(
    embedder: SentenceTransformerEmbedding,
) -> None:
    vector = embedder.embed_text(
        "India provides incentives for semiconductor manufacturing."
    )

    assert len(vector) == embedder.dimension
    assert all(np.isfinite(value) for value in vector)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason=(
        "Set RUN_PHASE2_EMBEDDING_TESTS=1 "
        "to run real BGE-M3 tests"
    ),
)
def test_bge_m3_batch_embeddings(
    embedder: SentenceTransformerEmbedding,
) -> None:
    texts = [
        "India semiconductor manufacturing policy.",
        "Electric vehicle incentives in India.",
        "Global semiconductor supply chains.",
    ]

    vectors = embedder.embed_texts(texts)

    assert len(vectors) == len(texts)

    assert all(
        len(vector) == embedder.dimension
        for vector in vectors
    )

    assert all(
        np.isfinite(value)
        for vector in vectors
        for value in vector
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason=(
        "Set RUN_PHASE2_EMBEDDING_TESTS=1 "
        "to run real BGE-M3 tests"
    ),
)
def test_bge_m3_embeddings_are_normalized(
    embedder: SentenceTransformerEmbedding,
) -> None:
    vector = np.asarray(
        embedder.embed_text(
            "Semiconductor manufacturing incentives."
        ),
        dtype=np.float32,
    )

    norm = np.linalg.norm(vector)

    assert np.isclose(
        norm,
        1.0,
        atol=1e-4,
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason=(
        "Set RUN_PHASE2_EMBEDDING_TESTS=1 "
        "to run real BGE-M3 tests"
    ),
)
def test_bge_m3_multilingual_embeddings(
    embedder: SentenceTransformerEmbedding,
) -> None:
    texts = [
        "Semiconductor manufacturing incentives in India.",
        "भारत में सेमीकंडक्टर विनिर्माण के लिए प्रोत्साहन।",
        "భారతదేశంలో సెమీకండక్టర్ తయారీకి ప్రోత్సాహకాలు.",
    ]

    vectors = embedder.embed_texts(texts)

    assert len(vectors) == 3

    for vector in vectors:
        assert len(vector) == embedder.dimension
        assert np.all(np.isfinite(vector))

        norm = np.linalg.norm(
            np.asarray(vector, dtype=np.float32)
        )

        assert np.isclose(
            norm,
            1.0,
            atol=1e-4,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE2_EMBEDDING_TESTS,
    reason=(
        "Set RUN_PHASE2_EMBEDDING_TESTS=1 "
        "to run real BGE-M3 tests"
    ),
)
def test_bge_m3_semantic_similarity(
    embedder: SentenceTransformerEmbedding,
) -> None:
    texts = [
        "India semiconductor manufacturing incentives.",
        "भारत में सेमीकंडक्टर विनिर्माण के लिए प्रोत्साहन।",
        "Electric vehicle incentives in India.",
    ]

    vectors = np.asarray(
        embedder.embed_texts(texts),
        dtype=np.float32,
    )

    similarity = vectors @ vectors.T

    # English and Hindi describe the same general concept.
    assert similarity[0, 1] > similarity[0, 2]
