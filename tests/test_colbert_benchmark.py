from src.evaluation.retrieval_benchmark import (
    RetrievalBenchmarkCase,
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
