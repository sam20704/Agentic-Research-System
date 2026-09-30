from pathlib import Path

from src.evaluation.retrieval_corpus import (
    build_retrieval_corpus,
)


def test_retrieval_corpus_is_deterministic():
    reference_dir = Path("data/references")

    first = build_retrieval_corpus(reference_dir)
    second = build_retrieval_corpus(reference_dir)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]

    assert [chunk.text for chunk in first] == [
        chunk.text for chunk in second
    ]


def test_retrieval_corpus_preserves_provenance():
    chunks = build_retrieval_corpus(
        Path("data/references")
    )

    assert chunks

    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.document_id
        assert chunk.source
        assert chunk.page_numbers
        assert chunk.text.strip()


def test_retrieval_corpus_has_unique_chunk_ids():
    chunks = build_retrieval_corpus(
        Path("data/references")
    )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))