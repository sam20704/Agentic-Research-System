"""Integration tests for the Qwen3 reranker."""

from __future__ import annotations

import pytest

from src.retrieval.models import DocumentChunk, RetrievalResult
from src.retrieval.reranking.cross_encoder import CrossEncoderReranker


pytestmark = pytest.mark.integration


def _make_candidate(
    chunk_id: str,
    text: str,
    rank: int,
    score: float,
) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="integration-doc-001",
        text=text,
        page_numbers=(1,),
        source="integration_test.pdf",
        section="Test",
    )

    return RetrievalResult(
        chunk=chunk,
        score=score,
        rank=rank,
        retrieval_method="hybrid",
        metadata={
            "rrf_score": score,
        },
    )


@pytest.fixture(scope="module")
def qwen_reranker() -> CrossEncoderReranker:
    """Load the real Qwen reranker once for the integration module."""
    return CrossEncoderReranker(
        device="cuda",
        max_length=2048,
        batch_size=8,
    )


def test_qwen_reranker_loads_on_cuda(
    qwen_reranker: CrossEncoderReranker,
) -> None:
    """Verify the real Qwen reranker loads on CUDA."""

    assert qwen_reranker.device.type == "cuda"
    assert qwen_reranker.model_name == "Qwen/Qwen3-Reranker-0.6B"


def test_qwen_reranks_relevant_evidence_higher(
    qwen_reranker: CrossEncoderReranker,
) -> None:
    """Verify Qwen assigns higher relevance to directly relevant evidence."""

    query = "What is India's semiconductor policy?"

    candidates = [
        _make_candidate(
            "irrelevant",
            (
                "The Indian monsoon season influences agricultural production "
                "and rainfall patterns across several states."
            ),
            rank=1,
            score=0.10,
        ),
        _make_candidate(
            "relevant",
            (
                "India's semiconductor policy provides incentives for "
                "semiconductor manufacturing, packaging, and related "
                "electronics infrastructure."
            ),
            rank=2,
            score=0.09,
        ),
    ]

    results = qwen_reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "relevant"
    assert results[0].score > results[1].score


def test_qwen_reranker_preserves_provenance(
    qwen_reranker: CrossEncoderReranker,
) -> None:
    """Verify reranking does not modify document provenance."""

    candidates = [
        _make_candidate(
            "chunk-001",
            "India established incentives for semiconductor manufacturing.",
            rank=1,
            score=0.20,
        ),
        _make_candidate(
            "chunk-002",
            "Electric vehicles receive incentives under selected policies.",
            rank=2,
            score=0.19,
        ),
    ]

    original = {
        candidate.chunk.chunk_id: candidate.chunk
        for candidate in candidates
    }

    results = qwen_reranker.rerank(
        query="What incentives exist for semiconductor manufacturing?",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2

    for result in results:
        original_chunk = original[result.chunk.chunk_id]

        assert result.chunk is original_chunk
        assert result.chunk.chunk_id == original_chunk.chunk_id
        assert result.chunk.document_id == original_chunk.document_id
        assert result.chunk.page_numbers == original_chunk.page_numbers
        assert result.chunk.source == original_chunk.source
        assert result.chunk.section == original_chunk.section


def test_qwen_reranker_preserves_rrf_score(
    qwen_reranker: CrossEncoderReranker,
) -> None:
    """Verify the original hybrid/RRF score remains available."""

    candidates = [
        _make_candidate(
            "chunk-001",
            "India semiconductor manufacturing incentives.",
            rank=1,
            score=0.1234,
        ),
        _make_candidate(
            "chunk-002",
            "India electric vehicle policy incentives.",
            rank=2,
            score=0.0987,
        ),
    ]

    results = qwen_reranker.rerank(
        query="semiconductor manufacturing incentives in India",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2

    for result in results:
        assert "rrf_score" in result.metadata
        assert "reranker_score" in result.metadata
        assert result.metadata["rrf_score"] in {0.1234, 0.0987}
        assert result.metadata["reranker_score"] == result.score


def test_qwen_reranker_assigns_valid_final_ranks(
    qwen_reranker: CrossEncoderReranker,
) -> None:
    """Verify final RetrievalResult ranks are valid and sequential."""

    candidates = [
        _make_candidate(
            "chunk-001",
            "India semiconductor manufacturing policy.",
            rank=1,
            score=0.30,
        ),
        _make_candidate(
            "chunk-002",
            "India electronics manufacturing incentives.",
            rank=2,
            score=0.20,
        ),
        _make_candidate(
            "chunk-003",
            "Global semiconductor supply chain policy.",
            rank=3,
            score=0.10,
        ),
    ]

    results = qwen_reranker.rerank(
        query="India semiconductor manufacturing policy",
        candidates=candidates,
        top_k=3,
    )

    assert [result.rank for result in results] == [1, 2, 3]
    assert all(result.rank >= 1 for result in results)
    assert all(result.retrieval_method == "cross_encoder" for result in results)