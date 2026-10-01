import pytest

from src.evaluation.retrieval_benchmark import (
    RetrievalBenchmark,
    RetrievalBenchmarkCase,
)
from src.retrieval.models import DocumentChunk, RetrievalResult


def make_result(chunk_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id="doc1",
            text=f"chunk {chunk_id}",
            page_numbers=(1,),
            source="fixture.pdf",
        ),
        score=float(-rank),
        rank=rank,
        retrieval_method="fake",
    )


def test_benchmark_case_supports_multiple_relevant_chunks():
    case = RetrievalBenchmarkCase(
        query_id="1",
        query="semiconductor policy",
        relevant_chunk_ids=(
            "chunk-1",
            "chunk-2",
        ),
    )

    assert case.query_id == "1"
    assert case.relevant_chunk_ids == (
        "chunk-1",
        "chunk-2",
    )


def test_benchmark_case_from_dict():
    case = RetrievalBenchmarkCase.from_dict(
        {
            "query_id": "1",
            "query": "semiconductor policy",
            "relevant_chunk_ids": [
                "chunk-1",
                "chunk-2",
            ],
        }
    )

    assert case.relevant_chunk_ids == (
        "chunk-1",
        "chunk-2"
    )


def test_recall_at_5_counts_multiple_relevant_chunks():
    results = [
        make_result("chunk-1", 1),
        make_result("chunk-3", 2),
        make_result("chunk-2", 3),
        make_result("chunk-9", 4),
        make_result("chunk-8", 5),
    ]

    recall, hits = RetrievalBenchmark._recall(
        results,
        {"chunk-1", "chunk-2", "chunk-7", "chunk-8"},
        5,
    )

    assert hits == 3
    assert recall == pytest.approx(0.75)


def test_recall_at_10_is_zero_when_no_relevant_result():
    results = [
        make_result("chunk-1", 1),
        make_result("chunk-2", 2),
    ]

    recall, hits = RetrievalBenchmark._recall(
        results,
        {"chunk-7"},
        10,
    )

    assert hits == 0
    assert recall == 0.0


def test_recall_with_missing_ground_truth_is_zero():
    results = [make_result("chunk-1", 1)]

    recall, hits = RetrievalBenchmark._recall(
        results,
        set(),
        5,
    )

    assert hits == 0
    assert recall == 0.0


def test_mrr_returns_one_for_relevant_result_at_rank_one():
    results = [
        make_result("chunk-1", 1),
        make_result("chunk-2", 2),
    ]

    assert RetrievalBenchmark._mrr(
        results,
        {"chunk-1"},
    ) == 1.0


def test_mrr_returns_reciprocal_rank_for_first_relevant_result():
    results = [
        make_result("chunk-1", 1),
        make_result("chunk-2", 2),
        make_result("chunk-3", 3),
    ]

    assert RetrievalBenchmark._mrr(
        results,
        {"chunk-3"},
    ) == pytest.approx(1 / 3)


def test_mrr_returns_zero_when_relevant_result_is_outside_results():
    results = [
        make_result("chunk-1", 1),
        make_result("chunk-2", 2),
    ]

    assert RetrievalBenchmark._mrr(
        results,
        {"chunk-9"},
    ) == 0.0
