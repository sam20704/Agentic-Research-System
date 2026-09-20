
import pytest

from src.evaluation.retrieval_benchmark import RetrievalBenchmark
from src.retrieval.models import DocumentChunk, RetrievalResult


def chunk(chunk_id):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text="example",
        page_numbers=(1,),
        source="policy.pdf",
        metadata={},
    )


class FakeDenseRetriever:
    def retrieve(self, query, top_k):
        return [
            RetrievalResult(
                chunk=chunk("chunk1"),
                score=0.95,
                rank=1,
                retrieval_method="dense",
            ),
            RetrievalResult(
                chunk=chunk("chunk2"),
                score=0.90,
                rank=2,
                retrieval_method="dense",
            ),
        ]


class FakeColBERTRetriever:
    def retrieve(self, query, top_k):
        return [
            RetrievalResult(
                chunk=chunk("chunk1"),
                score=5.2,
                rank=1,
                retrieval_method="colbert",
            ),
            RetrievalResult(
                chunk=chunk("chunk3"),
                score=4.8,
                rank=2,
                retrieval_method="colbert",
            ),
        ]


def test_dense_metrics():
    benchmark = RetrievalBenchmark(
        FakeDenseRetriever(),
        FakeColBERTRetriever(),
    )

    result = benchmark.benchmark_dense(
        query="semiconductor policy",
        expected_chunk="chunk1",
    )

    assert result.method == "bge-m3"
    assert result.recall_at_5 == 1.0
    assert result.recall_at_10 == 1.0
    assert result.mrr == 1.0
    assert result.latency_ms >= 0


def test_colbert_metrics():
    benchmark = RetrievalBenchmark(
        FakeDenseRetriever(),
        FakeColBERTRetriever(),
    )

    result = benchmark.benchmark_colbert(
        query="semiconductor policy",
        expected_chunk="chunk1",
    )

    assert result.method == "colbert"
    assert result.recall_at_5 == 1.0
    assert result.recall_at_10 == 1.0
    assert result.mrr == 1.0
    assert result.latency_ms >= 0


def test_missing_chunk_returns_zero_metrics():
    benchmark = RetrievalBenchmark(
        FakeDenseRetriever(),
        FakeColBERTRetriever(),
    )

    result = benchmark.benchmark_colbert(
        query="unknown query",
        expected_chunk="missing_chunk",
    )

    assert result.recall_at_5 == 0.0
    assert result.recall_at_10 == 0.0
    assert result.mrr == 0.0