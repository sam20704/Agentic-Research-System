import pytest

from src.document.models import BoundingBox
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.models import DocumentChunk, RetrievalResult


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


def make_chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=f"Text for {chunk_id}",
        page_numbers=(3,),
        source="policy.pdf",
        bounding_boxes=(
            BoundingBox(
                x0=10,
                y0=20,
                x1=30,
                y1=40,
            ),
        ),
        element_ids=("element-1",),
        metadata={"section": "Semiconductors"},
    )


def bm25_result(chunk_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        chunk=make_chunk(chunk_id),
        score=float(10 - rank),
        rank=rank,
        retrieval_method="bm25",
    )


def dense_result(chunk_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        chunk=make_chunk(chunk_id),
        score=float(0.9 - rank * 0.1),
        rank=rank,
        retrieval_method="dense",
    )


# ------------------------------------------------------------------
# Existing RRF behavior
# ------------------------------------------------------------------


def test_duplicate_chunks_are_merged():
    rrf = ReciprocalRankFusion()

    results = rrf.fuse(
        [bm25_result("chunk1", 1)],
        [dense_result("chunk1", 1)],
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk1"
    assert results[0].retrieval_method == "hybrid"


def test_higher_rank_in_both_lists_wins():
    rrf = ReciprocalRankFusion()

    results = rrf.fuse(
        [
            bm25_result("chunk1", 1),
            bm25_result("chunk2", 2),
        ],
        [
            dense_result("chunk2", 1),
            dense_result("chunk1", 2),
        ],
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id in {"chunk1", "chunk2"}


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def test_invalid_k_raises_value_error():
    with pytest.raises(ValueError):
        ReciprocalRankFusion(k=0)


def test_invalid_top_k_raises_value_error():
    rrf = ReciprocalRankFusion()

    with pytest.raises(ValueError):
        rrf.fuse([], top_k=0)


def test_mixed_retrieval_methods_raise_error():
    rrf = ReciprocalRankFusion()

    mixed = [
        bm25_result("chunk1", 1),
        dense_result("chunk2", 2),
    ]

    with pytest.raises(ValueError):
        rrf.fuse(mixed)


# ------------------------------------------------------------------
# Empty rankings
# ------------------------------------------------------------------


def test_empty_rankings_return_empty_list():
    rrf = ReciprocalRankFusion()

    assert rrf.fuse([], []) == []


# ------------------------------------------------------------------
# Single retrieval source
# ------------------------------------------------------------------


def test_bm25_only_results_are_preserved():
    rrf = ReciprocalRankFusion()

    results = rrf.fuse(
        [
            bm25_result("chunk1", 1),
            bm25_result("chunk2", 2),
        ]
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk1"
    assert results[1].chunk.chunk_id == "chunk2"

    assert results[0].chunk.metadata["bm25_rank"] == 1
    assert results[0].chunk.metadata["dense_rank"] is None


def test_dense_only_results_are_preserved():
    rrf = ReciprocalRankFusion()

    results = rrf.fuse(
        [
            dense_result("chunk1", 1),
            dense_result("chunk2", 2),
        ]
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk1"
    assert results[1].chunk.chunk_id == "chunk2"

    assert results[0].chunk.metadata["dense_rank"] == 1
    assert results[0].chunk.metadata["bm25_rank"] is None


# ------------------------------------------------------------------
# Provenance preservation
# ------------------------------------------------------------------


def test_provenance_is_preserved():
    rrf = ReciprocalRankFusion()

    result = rrf.fuse(
        [bm25_result("chunk1", 1)],
        [dense_result("chunk1", 1)],
    )[0]

    metadata = result.chunk.metadata

    assert metadata["bm25_rank"] == 1
    assert metadata["dense_rank"] == 1
    assert metadata["retrieval_sources"] == ["bm25", "dense"]
    assert "rrf_score" in metadata

    # Provenance from DocumentChunk survives fusion.
    assert result.chunk.document_id == "doc1"
    assert result.chunk.page_numbers == (3,)
    assert result.chunk.element_ids == ("element-1",)

    bbox = result.chunk.bounding_boxes[0]
    assert bbox.x0 == 10
    assert bbox.y0 == 20
    assert bbox.x1 == 30
    assert bbox.y1 == 40


# ------------------------------------------------------------------
# Deterministic ordering
# ------------------------------------------------------------------


def test_tie_breaking_is_deterministic():
    rrf = ReciprocalRankFusion()

    results = rrf.fuse(
        [bm25_result("chunkA", 1)],
        [dense_result("chunkB", 1)],
    )

    chunk_ids = [result.chunk.chunk_id for result in results]

    assert chunk_ids == sorted(chunk_ids)