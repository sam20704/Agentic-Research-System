"""Tests for BM25 sparse retrieval."""
#test/test_bm25_retriever.py 

from __future__ import annotations

import pytest

from src.document.models import BoundingBox
from src.retrieval.models import DocumentChunk
from src.retrieval.sparse import BM25Retriever


def make_chunk(
    chunk_id: str,
    text: str,
    page: int = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        page_numbers=(page,),
        source="test.pdf",
        section="Test Section",
        bounding_boxes=(
            BoundingBox(
                x0=0,
                y0=0,
                x1=100,
                y1=100,
            ),
        ),
        element_ids=(f"element-{chunk_id}",),
        metadata={"test": True},
    )


def test_retrieves_relevant_chunk_first() -> None:
    chunks = [
        make_chunk(
            "chunk-1",
            "The electric vehicle policy provides financial incentives.",
        ),
        make_chunk(
            "chunk-2",
            "FAME-II provides subsidies for electric vehicles.",
        ),
        make_chunk(
            "chunk-3",
            "The company manufactures semiconductor equipment.",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("FAME-II electric vehicle")

    assert results
    assert results[0].chunk.chunk_id == "chunk-2"
    assert results[0].retrieval_method == "bm25"


def test_top_k_is_respected() -> None:
    chunks = [
        make_chunk(f"chunk-{index}", "electric vehicle policy")
        for index in range(10)
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("electric vehicle", top_k=3)

    assert len(results) == 3


def test_ranks_are_sequential() -> None:
    chunks = [
        make_chunk("chunk-1", "electric vehicle policy"),
        make_chunk("chunk-2", "electric vehicle subsidy"),
        make_chunk("chunk-3", "electric vehicle charging"),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("electric vehicle")

    assert [result.rank for result in results] == list(
        range(1, len(results) + 1)
    )


def test_provenance_is_preserved() -> None:
    chunk = make_chunk(
        "chunk-1",
        "TSMC semiconductor manufacturing information",
        page=7,
    )

    retriever = BM25Retriever([chunk])

    results = retriever.search("TSMC")

    assert len(results) == 1

    result_chunk = results[0].chunk

    assert result_chunk.chunk_id == "chunk-1"
    assert result_chunk.document_id == "doc-001"
    assert result_chunk.page_numbers == (7,)
    assert result_chunk.source == "test.pdf"
    assert result_chunk.section == "Test Section"
    assert result_chunk.element_ids == ("element-chunk-1",)
    assert result_chunk.metadata == {"test": True}


def test_count() -> None:
    chunks = [
        make_chunk("chunk-1", "electric vehicle"),
        make_chunk("chunk-2", "FAME-II policy"),
    ]

    retriever = BM25Retriever(chunks)

    assert retriever.count() == 2


def test_add_chunks() -> None:
    first = make_chunk("chunk-1", "electric vehicle")
    second = make_chunk("chunk-2", "TSMC semiconductor")

    retriever = BM25Retriever([first])

    retriever.add_chunks([second])

    assert retriever.count() == 2

    results = retriever.search("TSMC")

    assert results[0].chunk.chunk_id == "chunk-2"


def test_clear() -> None:
    chunk = make_chunk("chunk-1", "electric vehicle")

    retriever = BM25Retriever([chunk])

    retriever.clear()

    assert retriever.count() == 0
    assert retriever.search("electric vehicle") == []


def test_empty_query_rejected() -> None:
    retriever = BM25Retriever(
        [make_chunk("chunk-1", "electric vehicle")]
    )

    with pytest.raises(ValueError):
        retriever.search("")


def test_invalid_top_k_rejected() -> None:
    retriever = BM25Retriever(
        [make_chunk("chunk-1", "electric vehicle")]
    )

    with pytest.raises(ValueError):
        retriever.search("electric vehicle", top_k=0)


def test_invalid_k1_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(k1=-1)


def test_invalid_b_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(b=-0.1)

    with pytest.raises(ValueError):
        BM25Retriever(b=1.1)


def test_empty_chunks_rejected() -> None:
    retriever = BM25Retriever()

    with pytest.raises(ValueError):
        retriever.index([])


def test_unindexed_search_returns_empty() -> None:
    retriever = BM25Retriever()

    assert retriever.search("electric vehicle") == []


def test_no_match_returns_empty() -> None:
    chunks = [
        make_chunk("chunk-1", "electric vehicle policy"),
        make_chunk("chunk-2", "FAME-II subsidy"),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("quantum computing")

    assert results == []


def test_tokenizer_preserves_project_terms() -> None:
    tokens = BM25Retriever._tokenize(
        "FAME-II PLI 2024-25 TSMC ₹500"
    )

    assert "fame-ii" in tokens
    assert "pli" in tokens
    assert "2024-25" in tokens
    assert "tsmc" in tokens
    assert "₹" in tokens


def test_ranking_is_deterministic() -> None:
    chunks = [
        make_chunk("chunk-1", "electric vehicle policy"),
        make_chunk("chunk-2", "electric vehicle subsidy"),
        make_chunk("chunk-3", "electric vehicle charging"),
    ]

    retriever = BM25Retriever(chunks)

    first = [
        result.chunk.chunk_id
        for result in retriever.search("electric vehicle")
    ]

    second = [
        result.chunk.chunk_id
        for result in retriever.search("electric vehicle")
    ]

    assert first == second