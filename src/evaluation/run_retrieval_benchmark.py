"""Run the frozen Phase 3.1 BGE-M3 vs ColBERT retrieval benchmark.

This benchmark intentionally compares only:

    BGE-M3 dense retrieval
    vs
    ColBERT late-interaction retrieval

It does NOT use:

    - BM25
    - RRF
    - Qwen reranking
    - query routing
    - adaptive retrieval
    - agent orchestration

The benchmark uses the exact frozen retrieval corpus and retrieval
ground truth produced for Phase 3.1.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil

from src.document.models import BoundingBox
from src.evaluation.retrieval_benchmark import (
    RetrievalBenchmarkCase,
    RetrievalBenchmarkResult,
    RetrievalBenchmarkSummary,
    load_cases,
    save_results,
    save_summary,
)
from src.retrieval.colbert import ColBERTConfig, ColBERTRetriever
from src.retrieval.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from src.retrieval.hybrid import DenseRetriever
from src.retrieval.models import DocumentChunk
from src.retrieval.vectorstore import QdrantVectorStore


# ---------------------------------------------------------------------------
# Frozen benchmark contract
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "retrieval_corpus.json"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "retrieval_ground_truth.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "results"
)

EXPECTED_CORPUS_SIZE = 1773
EXPECTED_QUERY_COUNT = 20
EXPECTED_GT_PAIRS = 186

TOP_K = 10

BGE_MODEL = "BAAI/bge-m3"

# ColBERT configuration is taken from the Phase 3.1 implementation.
COLBERT_MODEL = "colbert-ir/colbertv2.0"


# ---------------------------------------------------------------------------
# Corpus reconstruction
# ---------------------------------------------------------------------------


def _bounding_boxes(raw_boxes: list[dict[str, Any]]) -> tuple[BoundingBox, ...]:
    """Reconstruct canonical BoundingBox objects."""

    return tuple(
        BoundingBox(
            x0=float(box["x0"]),
            y0=float(box["y0"]),
            x1=float(box["x1"]),
            y1=float(box["y1"]),
        )
        for box in raw_boxes
    )


def load_frozen_corpus() -> list[DocumentChunk]:
    """Load and reconstruct the exact frozen benchmark corpus."""

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Frozen retrieval corpus not found: {CORPUS_PATH}"
        )

    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            "retrieval_corpus.json must contain a JSON list."
        )

    chunks: list[DocumentChunk] = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError(
                "Every retrieval corpus entry must be a JSON object."
            )

        chunk = DocumentChunk(
            chunk_id=str(item["chunk_id"]),
            document_id=str(item["document_id"]),
            text=str(item["text"]),
            page_numbers=tuple(
                int(page)
                for page in item["page_numbers"]
            ),
            source=str(item["source"]),
            section=item.get("section"),
            bounding_boxes=_bounding_boxes(
                item.get("bounding_boxes", [])
            ),
            element_ids=tuple(
                str(element_id)
                for element_id in item.get("element_ids", [])
            ),
            metadata=dict(item.get("metadata", {})),
        )

        chunks.append(chunk)

    # The corpus was frozen deterministically. Do not silently reorder it.
    if len(chunks) != EXPECTED_CORPUS_SIZE:
        raise ValueError(
            "Frozen corpus size mismatch: "
            f"expected {EXPECTED_CORPUS_SIZE}, "
            f"found {len(chunks)}."
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Frozen corpus contains duplicate chunk_id values."
        )

    return chunks


# ---------------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------------


def validate_ground_truth(
    cases: list[RetrievalBenchmarkCase],
    chunks: list[DocumentChunk],
) -> None:
    """Validate the frozen GT against the frozen corpus."""

    if len(cases) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            "Ground-truth query count mismatch: "
            f"expected {EXPECTED_QUERY_COUNT}, "
            f"found {len(cases)}."
        )

    corpus_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    gt_pairs = 0
    query_ids: set[str] = set()

    for case in cases:
        if case.query_id in query_ids:
            raise ValueError(
                f"Duplicate ground-truth query_id: {case.query_id}"
            )

        query_ids.add(case.query_id)

        if not case.relevant_chunk_ids:
            raise ValueError(
                f"Query {case.query_id} has no relevant chunks."
            )

        for chunk_id in case.relevant_chunk_ids:
            if chunk_id not in corpus_ids:
                raise ValueError(
                    "Ground-truth chunk does not exist in the "
                    f"frozen corpus: query={case.query_id}, "
                    f"chunk_id={chunk_id}"
                )

        gt_pairs += len(case.relevant_chunk_ids)

    if gt_pairs != EXPECTED_GT_PAIRS:
        raise ValueError(
            "Ground-truth pair count mismatch: "
            f"expected {EXPECTED_GT_PAIRS}, "
            f"found {gt_pairs}."
        )

    print(
        f"  Ground-truth queries : {len(cases)}"
    )
    print(
        f"  Ground-truth pairs   : {gt_pairs}"
    )


# ---------------------------------------------------------------------------
# BGE-M3
# ---------------------------------------------------------------------------


def build_bge_retriever(
    chunks: list[DocumentChunk],
) -> tuple[DenseRetriever, QdrantVectorStore, SentenceTransformerEmbedding]:
    """Build the BGE-M3 dense-only retriever."""

    print()
    print("Building BGE-M3 dense index...")
    print(f"  Model  : {BGE_MODEL}")
    print(f"  Chunks : {len(chunks)}")

    embedder = SentenceTransformerEmbedding(
        model_name=BGE_MODEL,
    )

    vectors = embedder.embed_chunks(chunks)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "BGE-M3 returned a different number of vectors "
            "than corpus chunks."
        )

    if not vectors:
        raise RuntimeError("BGE-M3 produced no vectors.")

    dimension = len(vectors[0])

    if dimension != 1024:
        raise RuntimeError(
            "Unexpected BGE-M3 dense dimension: "
            f"expected 1024, found {dimension}."
        )

    store = QdrantVectorStore(
        collection_name="phase3_1_bge_m3_benchmark",
        vector_size=dimension,
        path=None,
    )

    store.upsert(chunks, vectors)

    count = store.count()

    if count != EXPECTED_CORPUS_SIZE:
        raise RuntimeError(
            "BGE-M3 Qdrant index size mismatch: "
            f"expected {EXPECTED_CORPUS_SIZE}, "
            f"found {count}."
        )

    retriever = DenseRetriever(
        vectorstore=store,
        embedder=embedder,
    )

    print(f"  Dimension : {dimension}")
    print(f"  Indexed   : {count}")

    return retriever, store, embedder


# ---------------------------------------------------------------------------
# ColBERT
# ---------------------------------------------------------------------------


def build_colbert_retriever(
    chunks: list[DocumentChunk],
) -> ColBERTRetriever:
    """Build the standalone ColBERT index."""

    print()
    print("Building ColBERT index...")
    print(f"  Model  : {COLBERT_MODEL}")
    print(f"  Chunks : {len(chunks)}")

    config = ColBERTConfig(
        model_name=COLBERT_MODEL,
    )

    retriever = ColBERTRetriever(
        config=config,
    )

    retriever.build_index(chunks)

    count = retriever.index_size()

    if count != EXPECTED_CORPUS_SIZE:
        raise RuntimeError(
            "ColBERT index size mismatch: "
            f"expected {EXPECTED_CORPUS_SIZE}, "
            f"found {count}."
        )

    print(f"  Indexed : {count}")

    return retriever


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def warm_up(
    bge: DenseRetriever,
    colbert: ColBERTRetriever,
    cases: list[RetrievalBenchmarkCase],
) -> None:
    """Warm both retrieval paths before measuring latency."""

    if not cases:
        raise ValueError("Cannot warm up without benchmark queries.")

    warmup_queries = cases[:2]

    print()
    print("Warming up both retrievers...")

    for case in warmup_queries:
        bge.retrieve(
            case.query,
            top_k=TOP_K,
        )

    for case in warmup_queries:
        colbert.retrieve(
            case.query,
            top_k=TOP_K,
        )

    print("  Warm-up complete.")


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(
    results: list,
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Compute multi-positive retrieval recall@k.

    Recall@k is:

        |retrieved relevant chunks in top-k|
        -----------------------------------
             |all relevant chunks|

    This is intentionally NOT binary hit-rate.
    """

    if not relevant_chunk_ids:
        return 0.0

    retrieved = {
        result.chunk.chunk_id
        for result in results[:k]
    }

    hits = retrieved & relevant_chunk_ids

    return len(hits) / len(relevant_chunk_ids)


def reciprocal_rank(
    results: list,
    relevant_chunk_ids: set[str],
) -> float:
    """Compute reciprocal rank using the first relevant result."""

    for rank, result in enumerate(results, start=1):
        if result.chunk.chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------------------------
# Per-method benchmark
# ---------------------------------------------------------------------------


def benchmark_method(
    method: str,
    retriever,
    cases: list[RetrievalBenchmarkCase],
) -> list[RetrievalBenchmarkResult]:
    """Benchmark one retrieval method over all frozen queries."""

    print()
    print("=" * 72)
    print(f"BENCHMARK: {method.upper()}")
    print("=" * 72)

    process = psutil.Process(os.getpid())

    results: list[RetrievalBenchmarkResult] = []

    for index, case in enumerate(cases, start=1):
        relevant = set(case.relevant_chunk_ids)

        before = time.perf_counter()

        retrieved = retriever.retrieve(
            case.query,
            top_k=TOP_K,
        )

        elapsed_ms = (
            time.perf_counter() - before
        ) * 1000.0

        memory_mb = (
            process.memory_info().rss
            / (1024 * 1024)
        )

        result = RetrievalBenchmarkResult(
            query_id=case.query_id,
            method=method,

            relevant_count=len(relevant),

            retrieved_at_5=len({
                result.chunk.chunk_id
                for result in retrieved[:5]
            } & relevant),

            retrieved_at_10=len({
                result.chunk.chunk_id
                for result in retrieved[:10]
            } & relevant),

            recall_at_5=recall_at_k(
                retrieved,
                relevant,
                5,
            ),

            recall_at_10=recall_at_k(
                retrieved,
                relevant,
                10,
            ),

            mrr=reciprocal_rank(
                retrieved,
                relevant,
            ),

            latency_ms=elapsed_ms,
            memory_mb=memory_mb,
        )

        results.append(result)

        print(
            f"[{index:02d}/{len(cases)}] "
            f"query={case.query_id:>2} "
            f"R@5={result.recall_at_5:.3f} "
            f"R@10={result.recall_at_10:.3f} "
            f"MRR={result.mrr:.3f} "
            f"latency={result.latency_ms:.1f}ms"
        )

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(
    method: str,
    results: list[RetrievalBenchmarkResult],
) -> RetrievalBenchmarkSummary:
    """Build aggregate summary for one retrieval method."""

    latencies = [
        result.latency_ms
        for result in results
    ]

    memories = [
        result.memory_mb
        for result in results
    ]

    peak_memory_mb = max(memories) if memories else 0.0

    memory_delta_mb = (
        max(memories) - min(memories)
        if memories
        else 0.0
    )

    ground_truth_pairs = sum(
        result.relevant_count
        for result in results
    )

    return RetrievalBenchmarkSummary(
        method=method,
        queries=len(results),
        ground_truth_pairs=ground_truth_pairs,

        recall_at_5=statistics.mean(
            result.recall_at_5
            for result in results
        ),

        recall_at_10=statistics.mean(
            result.recall_at_10
            for result in results
        ),

        mrr=statistics.mean(
            result.mrr
            for result in results
        ),

        mean_latency_ms=statistics.mean(
            latencies
        ),

        median_latency_ms=statistics.median(
            latencies
        ),

        peak_memory_mb=peak_memory_mb,
        memory_delta_mb=memory_delta_mb,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_benchmark_metadata(
    path: Path,
    cases: list[RetrievalBenchmarkCase],
) -> None:
    """Save reproducibility metadata."""

    payload = {
        "benchmark": "Phase 3.1 retrieval benchmark",
        "corpus": {
            "path": str(CORPUS_PATH),
            "chunks": EXPECTED_CORPUS_SIZE,
        },
        "ground_truth": {
            "path": str(GROUND_TRUTH_PATH),
            "queries": len(cases),
            "relevant_pairs": EXPECTED_GT_PAIRS,
        },
        "methods": [
            {
                "name": "bge-m3",
                "model": BGE_MODEL,
                "retrieval_type": "dense-only",
            },
            {
                "name": "colbert",
                "model": COLBERT_MODEL,
                "retrieval_type": "late-interaction",
            },
        ],
        "top_k": TOP_K,
        "excluded_components": [
            "BM25",
            "RRF",
            "Qwen reranker",
            "query router",
            "adaptive routing",
            "agent orchestration",
        ],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("PHASE 3.1 — FROZEN RETRIEVAL BENCHMARK")
    print("BGE-M3 dense vs ColBERT late interaction")
    print("=" * 72)

    print()
    print("[1/7] Loading frozen retrieval corpus...")

    chunks = load_frozen_corpus()

    print(
        f"  Frozen corpus : {len(chunks)} chunks"
    )

    print()
    print("[2/7] Loading frozen retrieval ground truth...")

    cases = load_cases(GROUND_TRUTH_PATH)

    print(
        f"  Queries       : {len(cases)}"
    )

    print()
    print("[3/7] Validating frozen ground truth...")

    validate_ground_truth(
        cases,
        chunks,
    )

    print()
    print("[4/7] Building BGE-M3 dense-only index...")

    bge, bge_store, bge_embedder = build_bge_retriever(
        chunks
    )

    print()
    print("[5/7] Building ColBERT-only index...")

    colbert = build_colbert_retriever(
        chunks
    )

    print()
    print("[6/7] Warming up and benchmarking...")

    warm_up(
        bge,
        colbert,
        cases,
    )

    bge_results = benchmark_method(
        "bge-m3",
        bge,
        cases,
    )

    colbert_results = benchmark_method(
        "colbert",
        colbert,
        cases,
    )

    all_results = (
        bge_results
        + colbert_results
    )

    summaries = [
        build_summary(
            "bge-m3",
            bge_results,
        ),
        build_summary(
            "colbert",
            colbert_results,
        ),
    ]

    print()
    print("[7/7] Saving benchmark results...")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        RESULTS_DIR
        / "phase3_1_retrieval_results.json"
    )

    summary_path = (
        RESULTS_DIR
        / "phase3_1_retrieval_summary.json"
    )

    metadata_path = (
        RESULTS_DIR
        / "phase3_1_retrieval_metadata.json"
    )

    save_results(
        results_path,
        all_results,
    )

    save_summary(
        summary_path,
        summaries,
    )

    save_benchmark_metadata(
        metadata_path,
        cases,
    )

    print()
    print("=" * 72)
    print("PHASE 3.1 BENCHMARK COMPLETE")
    print("=" * 72)

    for summary in summaries:
        print()
        print(f"{summary.method}")
        print(
            f"  Recall@5      : "
            f"{summary.recall_at_5:.4f}"
        )
        print(
            f"  Recall@10     : "
            f"{summary.recall_at_10:.4f}"
        )
        print(
            f"  MRR           : "
            f"{summary.mrr:.4f}"
        )
        print(
            f"  Mean latency  : "
            f"{summary.mean_latency_ms:.2f} ms"
        )
        print(
            f"  Median latency: "
            f"{summary.median_latency_ms:.2f} ms"
        )
    print()
    print(f"Results : {results_path}")
    print(f"Summary : {summary_path}")
    print(f"Metadata: {metadata_path}")

    # Explicitly release the Qdrant-backed BGE store.
    #
    # This does not affect saved benchmark results.
    try:
        bge_store.close()
    except AttributeError:
        pass

    del bge_embedder


if __name__ == "__main__":
    main()