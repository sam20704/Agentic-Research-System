
from __future__ import annotations

import time
from dataclasses import dataclass

import psutil

from src.retrieval.colbert.retriever import ColBERTRetriever
from src.retrieval.hybrid import DenseRetriever
from src.retrieval.models import DocumentChunk
from src.retrieval.vectorstore import QdrantVectorStore


@dataclass(frozen=True)
class RetrievalBenchmarkResult:
    method: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    latency_ms: float
    memory_mb: float


class RetrievalBenchmark:
    """
    Benchmark ColBERT against the existing BGE-M3 dense baseline.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        colbert_retriever: ColBERTRetriever,
    ):
        self.dense = dense_retriever
        self.colbert = colbert_retriever

    @staticmethod
    def _recall(results, expected_chunk, k):
        ids = [r.chunk.chunk_id for r in results[:k]]
        return float(expected_chunk in ids)

    @staticmethod
    def _mrr(results, expected_chunk):
        for idx, result in enumerate(results, start=1):
            if result.chunk.chunk_id == expected_chunk:
                return 1.0 / idx
        return 0.0

    @staticmethod
    def _memory_mb():
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def benchmark_dense(
        self,
        query: str,
        expected_chunk: str,
        top_k: int = 10,
    ) -> RetrievalBenchmarkResult:
        start_memory = self._memory_mb()
        start = time.perf_counter()

        results = self.dense.retrieve(
            query=query,
            top_k=top_k,
        )

        latency = (time.perf_counter() - start) * 1000

        return RetrievalBenchmarkResult(
            method="bge-m3",
            recall_at_5=self._recall(results, expected_chunk, 5),
            recall_at_10=self._recall(results, expected_chunk, 10),
            mrr=self._mrr(results, expected_chunk),
            latency_ms=latency,
            memory_mb=max(self._memory_mb() - start_memory, 0.0),
        )

    def benchmark_colbert(
        self,
        query: str,
        expected_chunk: str,
        top_k: int = 10,
    ) -> RetrievalBenchmarkResult:
        start_memory = self._memory_mb()
        start = time.perf_counter()

        results = self.colbert.retrieve(
            query=query,
            top_k=top_k,
        )

        latency = (time.perf_counter() - start) * 1000

        return RetrievalBenchmarkResult(
            method="colbert",
            recall_at_5=self._recall(results, expected_chunk, 5),
            recall_at_10=self._recall(results, expected_chunk, 10),
            mrr=self._mrr(results, expected_chunk),
            latency_ms=latency,
            memory_mb=max(self._memory_mb() - start_memory, 0.0),
        )