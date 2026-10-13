"""Benchmark hierarchical retrieval against the frozen retrieval corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.document.models import BoundingBox
from src.retrieval.hierarchical import (
    HierarchicalRetrievalConfig,
    HierarchicalRetriever,
)
from src.retrieval.models import DocumentChunk


DEFAULT_CORPUS = Path("data/benchmarks/retrieval_corpus.json")
DEFAULT_GROUND_TRUTH = Path("data/benchmarks/retrieval_ground_truth.json")
DEFAULT_OUTPUT = Path(
    "data/benchmarks/phase3_3_hierarchical_results.json"
)


def load_json(path: Path) -> Any:
    """Load JSON from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_chunks(path: Path) -> List[DocumentChunk]:
    """Load frozen benchmark chunks."""
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list of chunks in {path}"
        )

    chunks: list[DocumentChunk] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Corpus entry {index} in {path} must be an object")
        raw_boxes = item.get("bounding_boxes", [])
        boxes = tuple(
            box if isinstance(box, BoundingBox) else BoundingBox(
                x0=float(box["x0"]), y0=float(box["y0"]),
                x1=float(box["x1"]), y1=float(box["y1"]),
            )
            for box in raw_boxes
        )
        chunks.append(DocumentChunk(
            chunk_id=str(item["chunk_id"]),
            document_id=str(item["document_id"]),
            text=str(item["text"]),
            page_numbers=tuple(int(page) for page in item["page_numbers"]),
            source=str(item["source"]),
            section=item.get("section"),
            bounding_boxes=boxes,
            element_ids=tuple(str(element_id) for element_id in item.get("element_ids", [])),
            metadata=dict(item.get("metadata", {})),
        ))
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"Duplicate chunk_id values found in {path}")
    return chunks


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """Load frozen benchmark queries."""
    data = load_json(path)

    if isinstance(data, dict):
        for key in (
            "queries",
            "ground_truth",
            "cases",
            "items",
        ):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            f"Could not find query list in {path}"
        )

    return data


def query_text(item: Dict[str, Any]) -> str:
    """Extract the query text from a benchmark item."""
    for key in ("query", "question", "text"):
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        f"Could not find query text in benchmark item: {item}"
    )


def ground_truth_ids(item: Dict[str, Any]) -> set[str]:
    """Extract relevant chunk IDs from a benchmark item."""
    candidates = (
        item.get("relevant_chunk_ids"),
        item.get("ground_truth_chunk_ids"),
        item.get("relevant_ids"),
        item.get("chunk_ids"),
        item.get("ground_truth"),
    )

    for value in candidates:
        if not isinstance(value, list):
            continue

        result = set()

        for entry in value:
            if isinstance(entry, str):
                result.add(entry)
            elif isinstance(entry, dict):
                chunk_id = entry.get("chunk_id")

                if isinstance(chunk_id, str):
                    result.add(chunk_id)

        if result:
            return result

    return set()


def recall_at_k(
    result_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate recall at k."""
    if not relevant_ids:
        return 0.0

    retrieved = set(result_ids[:k])

    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(
    result_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    """Calculate reciprocal rank."""
    if not relevant_ids:
        return 0.0

    for rank, chunk_id in enumerate(result_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def benchmark(
    corpus_path: Path,
    ground_truth_path: Path,
    output_path: Path,
    top_k: int,
) -> Dict[str, Any]:
    """Run the hierarchical retrieval benchmark."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    chunks = load_chunks(corpus_path)
    queries = load_queries(ground_truth_path)
    if not queries:
        raise ValueError(f"Ground-truth file contains no queries: {ground_truth_path}")
    if not chunks:
        raise ValueError(f"Corpus file contains no chunks: {corpus_path}")
    corpus_ids = {chunk.chunk_id for chunk in chunks}
    for index, item in enumerate(queries):
        relevant = ground_truth_ids(item)
        if not relevant:
            raise ValueError(f"Ground-truth query at index {index} has no relevant chunk IDs")
        unknown = relevant - corpus_ids
        if unknown:
            raise ValueError(f"Ground-truth query at index {index} references unknown chunk IDs: {sorted(unknown)}")

    config = HierarchicalRetrievalConfig(
        top_sections=5,
        top_pages_per_section=3,
        top_chunks_per_page=10,
        top_k=top_k,
    )

    tracemalloc.start()

    index_start = time.perf_counter()

    retriever = HierarchicalRetriever(
        chunks,
        config=config,
    )

    index_time_ms = (
        time.perf_counter() - index_start
    ) * 1000

    _, index_peak = tracemalloc.get_traced_memory()

    latencies_ms: List[float] = []
    recall5: List[float] = []
    recall10: List[float] = []
    reciprocal_ranks: List[float] = []

    section_counts: List[int] = []
    page_counts: List[int] = []
    chunk_counts: List[int] = []
    section_scored_counts: List[int] = []
    page_scored_counts: List[int] = []
    chunk_scored_counts: List[int] = []

    per_query: List[Dict[str, Any]] = []

    for item in queries:
        query = query_text(item)
        relevant_ids = ground_truth_ids(item)

        start = time.perf_counter()

        results = retriever.retrieve(
            query,
            top_k=top_k,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        result_ids = [
            result.chunk.chunk_id
            for result in results
        ]

        stats = retriever.last_stats

        latencies_ms.append(latency_ms)
        recall5.append(
            recall_at_k(
                result_ids,
                relevant_ids,
                5,
            )
        )
        recall10.append(
            recall_at_k(
                result_ids,
                relevant_ids,
                10,
            )
        )
        reciprocal_ranks.append(
            reciprocal_rank(
                result_ids,
                relevant_ids,
            )
        )

        section_counts.append(
            stats.section_candidates
        )
        page_counts.append(
            stats.page_candidates
        )
        chunk_counts.append(stats.chunk_candidates)
        section_scored_counts.append(stats.section_nodes_scored)
        page_scored_counts.append(stats.page_nodes_scored)
        chunk_scored_counts.append(stats.chunk_nodes_scored)

        per_query.append(
            {
                "query": query,
                "relevant_chunk_ids": sorted(
                    relevant_ids
                ),
                "retrieved_chunk_ids": result_ids,
                "recall_at_5": recall5[-1],
                "recall_at_10": recall10[-1],
                "reciprocal_rank": reciprocal_ranks[-1],
                "latency_ms": latency_ms,
                "section_candidates": (
                    stats.section_candidates
                ),
                "page_candidates": (
                    stats.page_candidates
                ),
                "chunk_candidates": stats.chunk_candidates,
                "section_nodes_scored": stats.section_nodes_scored,
                "page_nodes_scored": stats.page_nodes_scored,
                "chunk_nodes_scored": stats.chunk_nodes_scored,
            }
        )

    current_memory, peak_memory = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    result = {
        "phase": "3.3",
        "method": "hierarchical_retrieval",
        "runtime": {
            "model": "deterministic / none",
            "device": "CPU",
            "batch_size": None,
            "memory_metric": "Python allocations measured by tracemalloc; not total process RSS or VRAM",
        },
        "index": {
            "nodes": len(retriever.index.nodes),
        },
        "corpus": {
            "path": str(corpus_path),
            "chunks": len(chunks),
        },
        "ground_truth": {
            "path": str(ground_truth_path),
            "queries": len(queries),
        },
        "configuration": {
            "top_sections": config.top_sections,
            "top_pages_per_section": (
                config.top_pages_per_section
            ),
            "top_chunks_per_page": (
                config.top_chunks_per_page
            ),
            "top_k": top_k,
        },
        "metrics": {
            "recall_at_5": (
                statistics.mean(recall5)
                if recall5
                else 0.0
            ),
            "recall_at_10": (
                statistics.mean(recall10)
                if recall10
                else 0.0
            ),
            "mrr": (
                statistics.mean(reciprocal_ranks)
                if reciprocal_ranks
                else 0.0
            ),
            "mean_latency_ms": (
                statistics.mean(latencies_ms)
                if latencies_ms
                else 0.0
            ),
            "median_latency_ms": (
                statistics.median(latencies_ms)
                if latencies_ms
                else 0.0
            ),
            "index_construction_ms": index_time_ms,
            "current_memory_bytes": current_memory,
            "peak_memory_bytes": peak_memory,
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "memory_delta_mb": max(
                0.0,
                (peak_memory - index_peak)
                / (1024 * 1024),
            ),
            "average_section_candidates": (
                statistics.mean(section_counts)
                if section_counts
                else 0.0
            ),
            "average_page_candidates": (
                statistics.mean(page_counts)
                if page_counts
                else 0.0
            ),
            "average_chunk_candidates": (
                statistics.mean(chunk_counts) if chunk_counts else 0.0
            ),
            "average_section_nodes_scored": (
                statistics.mean(section_scored_counts) if section_scored_counts else 0.0
            ),
            "average_page_nodes_scored": (
                statistics.mean(page_scored_counts) if page_scored_counts else 0.0
            ),
            "average_chunk_nodes_scored": (
                statistics.mean(chunk_scored_counts) if chunk_scored_counts else 0.0
            ),
            "average_chunk_candidate_reduction_ratio": (
                1.0 - statistics.mean(chunk_counts) / len(chunks)
                if chunk_counts and chunks else 0.0
            ),
        },
        "per_query": per_query,
        "baseline_comparison": {
            "status": "not_run",
            "reason": (
                "A frozen Phase 2 result artifact using "
                "the same 20-query benchmark was not "
                "available in the supplied project files."
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Phase 3.3 hierarchical retrieval."
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main() -> None:
    """Run the benchmark CLI."""
    args = parse_args()

    result = benchmark(
        corpus_path=args.corpus,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        top_k=args.top_k,
    )

    metrics = result["metrics"]

    print("Phase 3.3 Hierarchical Retrieval Benchmark")
    print("-------------------------------------------")
    print(
        f"Corpus chunks: {result['corpus']['chunks']}"
    )
    print(
        f"Queries: {result['ground_truth']['queries']}"
    )
    print(
        f"Recall@5: {metrics['recall_at_5']:.4f}"
    )
    print(
        f"Recall@10: {metrics['recall_at_10']:.4f}"
    )
    print(
        f"MRR: {metrics['mrr']:.4f}"
    )
    print(
        "Mean latency: "
        f"{metrics['mean_latency_ms']:.2f} ms"
    )
    print(
        "Median latency: "
        f"{metrics['median_latency_ms']:.2f} ms"
    )
    print(
        "Index construction: "
        f"{metrics['index_construction_ms']:.2f} ms"
    )
    print(
        "Average section candidates: "
        f"{metrics['average_section_candidates']:.2f}"
    )
    print(
        "Average page candidates: "
        f"{metrics['average_page_candidates']:.2f}"
    )
    print(
        "Average chunk candidates: "
        f"{metrics['average_chunk_candidates']:.2f}"
    )
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()