"""Dedicated Phase 3.1 retrieval benchmark.

Compares:
    - BGE-M3 dense retrieval
    - ColBERT-v2 late-interaction retrieval

Benchmark constraints:
    - Frozen 1,773-chunk corpus
    - Frozen 20-query retrieval ground truth
    - 186 relevant query/chunk pairs
    - No BM25
    - No RRF
    - No reranker
    - No router
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable, Sequence

import psutil

from src.retrieval.models import RetrievalResult


EXPECTED_CORPUS_SIZE = 1773
EXPECTED_QUERY_COUNT = 20
EXPECTED_GT_PAIRS = 186
EXCLUDED_QUERY_IDS = {"11", "12", "13", "14", "15"}

DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    """One frozen retrieval benchmark query."""

    query_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "RetrievalBenchmarkCase":

        if "query_id" not in data:
            raise ValueError("Missing query_id.")

        if "query" not in data:
            raise ValueError(
                f"Query {data['query_id']} is missing query text."
            )

        relevant = data.get("relevant_chunk_ids")

        if not isinstance(relevant, list):
            raise ValueError(
                f"Query {data['query_id']} must contain "
                "'relevant_chunk_ids'."
            )

        if not relevant:
            raise ValueError(
                f"Query {data['query_id']} has no relevant chunks."
            )

        query_id = str(data["query_id"])

        if query_id in EXCLUDED_QUERY_IDS:
            raise ValueError(
                f"Excluded query {query_id} must not appear "
                "in retrieval ground truth."
            )

        unique_relevant = tuple(
            dict.fromkeys(
                str(chunk_id)
                for chunk_id in relevant
            )
        )

        if len(unique_relevant) != len(relevant):
            raise ValueError(
                f"Query {query_id} contains duplicate relevant chunk IDs."
            )

        return cls(
            query_id=query_id,
            query=str(data["query"]),
            relevant_chunk_ids=unique_relevant,
        )


@dataclass(frozen=True)
class RetrievalBenchmarkResult:
    """Per-query benchmark result."""

    query_id: str
    method: str

    relevant_count: int
    retrieved_at_5: int
    retrieved_at_10: int

    recall_at_5: float
    recall_at_10: float
    mrr: float

    latency_ms: float
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0



@dataclass(frozen=True)
class RetrievalBenchmarkSummary:
    """Aggregate benchmark result."""

    method: str
    queries: int
    ground_truth_pairs: int

    recall_at_5: float
    recall_at_10: float
    mrr: float

    mean_latency_ms: float
    median_latency_ms: float

    peak_memory_mb: float
    memory_delta_mb: float
    peak_gpu_memory_mb: float = 0.0
    indexing_time_ms: float = 0.0


class RetrievalBenchmark:
    """Evaluate one retrieval implementation."""

    def __init__(
        self,
        method: str,
        retrieve_fn: Callable[
            [str, int],
            list[RetrievalResult],
        ],
        peak_memory_mb: float,
        baseline_memory_mb: float,
    ) -> None:

        if not method:
            raise ValueError("method must not be empty.")

        self.method = method
        self.retrieve_fn = retrieve_fn
        self.peak_memory_mb = peak_memory_mb
        self.baseline_memory_mb = baseline_memory_mb

    @staticmethod
    def _recall(
        results: Sequence[RetrievalResult],
        relevant_chunk_ids: Iterable[str],
        k: int,
    ) -> tuple[float, int]:

        relevant = set(relevant_chunk_ids)

        if not relevant:
            return 0.0, 0

        retrieved = {
            result.chunk.chunk_id
            for result in results[:k]
        }

        hits = len(
            retrieved.intersection(relevant)
        )

        recall = hits / len(relevant)

        return recall, hits

    @staticmethod
    def _mrr(
        results: Sequence[RetrievalResult],
        relevant_chunk_ids: Iterable[str],
    ) -> float:

        relevant = set(relevant_chunk_ids)

        for rank, result in enumerate(
            results,
            start=1,
        ):
            if result.chunk.chunk_id in relevant:
                return 1.0 / rank

        return 0.0

    def benchmark_case(
        self,
        case: RetrievalBenchmarkCase,
        top_k: int = DEFAULT_TOP_K,
    ) -> RetrievalBenchmarkResult:

        if top_k < 10:
            raise ValueError(
                "top_k must be at least 10 because "
                "Recall@10 is required."
            )

        start = time.perf_counter()

        results = self.retrieve_fn(
            case.query,
            top_k,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        recall_at_5, retrieved_at_5 = self._recall(
            results,
            case.relevant_chunk_ids,
            5,
        )

        recall_at_10, retrieved_at_10 = self._recall(
            results,
            case.relevant_chunk_ids,
            10,
        )

        mrr = self._mrr(
            results,
            case.relevant_chunk_ids,
        )

        return RetrievalBenchmarkResult(
            query_id=case.query_id,
            method=self.method,
            relevant_count=len(
                case.relevant_chunk_ids
            ),
            retrieved_at_5=retrieved_at_5,
            retrieved_at_10=retrieved_at_10,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            latency_ms=latency_ms,
        )

    def benchmark_cases(
        self,
        cases: Sequence[RetrievalBenchmarkCase],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievalBenchmarkResult]:

        if len(cases) != EXPECTED_QUERY_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_QUERY_COUNT} queries, "
                f"got {len(cases)}."
            )

        results = []

        for case in cases:
            results.append(
                self.benchmark_case(
                    case,
                    top_k=top_k,
                )
            )

        return results

    def summary(
        self,
        results: Sequence[RetrievalBenchmarkResult],
    ) -> RetrievalBenchmarkSummary:

        if not results:
            raise ValueError(
                "Cannot summarize empty results."
            )

        if len(results) != EXPECTED_QUERY_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_QUERY_COUNT} "
                f"results, got {len(results)}."
            )

        return RetrievalBenchmarkSummary(
            method=self.method,
            queries=len(results),
            ground_truth_pairs=sum(
                result.relevant_count
                for result in results
            ),
            recall_at_5=mean(
                result.recall_at_5
                for result in results
            ),
            recall_at_10=mean(
                result.recall_at_10
                for result in results
            ),
            mrr=mean(
                result.mrr
                for result in results
            ),
            mean_latency_ms=mean(
                result.latency_ms
                for result in results
            ),
            median_latency_ms=median(
                result.latency_ms
                for result in results
            ),
            peak_memory_mb=self.peak_memory_mb,
            memory_delta_mb=(
                self.peak_memory_mb
                - self.baseline_memory_mb
            ),
        )


def load_cases(
    path: str | Path,
) -> list[RetrievalBenchmarkCase]:

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            "Retrieval ground truth must be a JSON list."
        )

    cases = [
        RetrievalBenchmarkCase.from_dict(item)
        for item in data
    ]

    if len(cases) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUERY_COUNT} retrieval queries, "
            f"got {len(cases)}."
        )

    total_pairs = sum(
        len(case.relevant_chunk_ids)
        for case in cases
    )

    if total_pairs != EXPECTED_GT_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_GT_PAIRS} GT pairs, "
            f"got {total_pairs}."
        )

    return cases


def validate_ground_truth_against_corpus(
    cases: Sequence[RetrievalBenchmarkCase],
    corpus_chunk_ids: Iterable[str],
) -> None:

    corpus_ids = set(corpus_chunk_ids)

    if len(corpus_ids) != EXPECTED_CORPUS_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_CORPUS_SIZE} unique corpus chunks, "
            f"got {len(corpus_ids)}."
        )

    missing = set()

    for case in cases:
        missing.update(
            chunk_id
            for chunk_id in case.relevant_chunk_ids
            if chunk_id not in corpus_ids
        )

    if missing:
        preview = sorted(missing)[:10]

        raise ValueError(
            "Ground truth contains chunk IDs that are not "
            f"present in the frozen corpus. Examples: {preview}"
        )


def current_memory_mb() -> float:
    """Return current process RSS in MB."""

    process = psutil.Process()

    return (
        process.memory_info().rss
        / (1024 * 1024)
    )


def save_results(
    path: str | Path,
    results: Sequence[RetrievalBenchmarkResult],
) -> None:

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            [
                asdict(result)
                for result in results
            ],
            handle,
            indent=2,
        )


def save_summary(
    path: str | Path,
    summary: RetrievalBenchmarkSummary,
) -> None:

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            asdict(summary),
            handle,
            indent=2,
        )


def save_summaries(
    path: str | Path,
    summaries: Sequence[RetrievalBenchmarkSummary],
) -> None:
    """Save aggregate summaries for multiple benchmark methods."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "corpus_size": EXPECTED_CORPUS_SIZE,
        "query_count": EXPECTED_QUERY_COUNT,
        "ground_truth_pairs": EXPECTED_GT_PAIRS,
        "methods": {
            summary.method: asdict(summary)
            for summary in summaries
        },
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


__all__ = [
    "EXPECTED_CORPUS_SIZE",
    "EXPECTED_QUERY_COUNT",
    "EXPECTED_GT_PAIRS",
    "EXCLUDED_QUERY_IDS",
    "RetrievalBenchmarkCase",
    "RetrievalBenchmarkResult",
    "RetrievalBenchmarkSummary",
    "RetrievalBenchmark",
    "load_cases",
    "validate_ground_truth_against_corpus",
    "current_memory_mb",
    "save_results",
    "save_summary",
    "save_summaries",
]
