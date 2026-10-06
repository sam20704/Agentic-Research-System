"""Phase 3.2 Qwen PageIndex navigation evaluation.

This evaluates the Qwen navigator against the exact same ground-truth cases
used by the deterministic PageIndex navigation benchmark.

Metrics:
    - Recall@5
    - MRR
    - mean latency

The evaluator does not modify the PageIndex tree or deterministic navigator.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.retrieval.pageindex.qwen_navigator import (
    QwenPageIndexNavigator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDEX_PATH = (
    PROJECT_ROOT
    / ".pageindex"
    / "Global_Supply_Chain_Report_DEC2025_EV_1.json"
)

DEFAULT_CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "pageindex_navigation_cases.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "results"
    / "phase3_2_pageindex_qwen_navigation_results.json"
)

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reciprocal_rank(
    retrieved_titles: list[str],
    expected_titles: list[str],
) -> float:
    expected = set(expected_titles)

    for rank, title in enumerate(
        retrieved_titles,
        start=1,
    ):
        if title in expected:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    retrieved_titles: list[str],
    expected_titles: list[str],
) -> float:
    expected = set(expected_titles)

    if not expected:
        return 0.0

    matched = expected.intersection(retrieved_titles)

    return len(matched) / len(expected)


def evaluate_case(
    navigator: QwenPageIndexNavigator,
    case: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    query = str(case["query"])

    expected_titles = [
        str(title)
        for title in case["expected_titles"]
    ]

    started = time.perf_counter()

    candidates = navigator.navigate(
        query=query,
        top_k=top_k,
    )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000.0

    retrieved_titles = [
        str(candidate.node["title"])
        for candidate in candidates
    ]

    retrieved_node_ids = [
        str(candidate.node["node_id"])
        for candidate in candidates
    ]

    retrieved_paths = [
        list(candidate.path)
        for candidate in candidates
    ]

    matched_expected_titles = [
        title
        for title in expected_titles
        if title in retrieved_titles
    ]

    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "query": query,
        "expected_titles": expected_titles,
        "top_k": top_k,
        "latency_ms": elapsed_ms,
        "retrieved_titles": retrieved_titles,
        "retrieved_node_ids": retrieved_node_ids,
        "retrieved_paths": retrieved_paths,
        "matched_expected_titles": matched_expected_titles,
        "recall_at_k": recall_at_k(
            retrieved_titles,
            expected_titles,
        ),
        "first_relevant_rank": (
            next(
                (
                    index
                    for index, title in enumerate(
                        retrieved_titles,
                        start=1,
                    )
                    if title in set(expected_titles)
                ),
                None,
            )
        ),
        "mrr": reciprocal_rank(
            retrieved_titles,
            expected_titles,
        ),
    }


def aggregate(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise ValueError("No evaluation results")

    return {
        "cases": len(results),
        "recall_at_k": sum(
            result["recall_at_k"]
            for result in results
        ) / len(results),
        "mrr": sum(
            result["mrr"]
            for result in results
        ) / len(results),
        "mean_latency_ms": sum(
            result["latency_ms"]
            for result in results
        ) / len(results),
    }


def aggregate_by_category(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for result in results:
        grouped[result["category"]].append(result)

    return {
        category: aggregate(category_results)
        for category, category_results in sorted(
            grouped.items()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen-based PageIndex navigation."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )

    parser.add_argument(
        "--cases-path",
        type=Path,
        default=DEFAULT_CASES_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    if not args.index_path.exists():
        raise SystemExit(
            f"PageIndex index not found: {args.index_path}"
        )

    if not args.cases_path.exists():
        raise SystemExit(
            f"Evaluation cases not found: {args.cases_path}"
        )

    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    index = load_json(args.index_path)
    cases = load_json(args.cases_path)

    if not isinstance(cases, list) or not cases:
        raise SystemExit(
            "Evaluation cases must be a non-empty JSON list"
        )

    navigator = QwenPageIndexNavigator(
        index,
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
        max_output_tokens=256,
    )

    results: list[dict[str, Any]] = []

    print("=" * 72)
    print("PHASE 3.2 — QWEN PAGEINDEX NAVIGATION EVALUATION")
    print("=" * 72)
    print(f"Navigator       : QwenPageIndexNavigator")
    print(f"Model           : {args.model}")
    print(f"Top-k           : {args.top_k}")
    print(f"Cases           : {len(cases)}")
    print()

    for case in cases:
        result = evaluate_case(
            navigator,
            case,
            top_k=args.top_k,
        )

        results.append(result)

        print(
            f"{result['case_id']}: "
            f"Recall={result['recall_at_k']:.4f}, "
            f"MRR={result['mrr']:.4f}, "
            f"Latency={result['latency_ms']:.2f} ms"
        )

    overall = aggregate(results)
    by_category = aggregate_by_category(results)

    print()
    print("OVERALL")
    print("-" * 72)
    print(
        f"Recall@{args.top_k}  : "
        f"{overall['recall_at_k']:.4f}"
    )
    print(
        f"MRR             : "
        f"{overall['mrr']:.4f}"
    )
    print(
        f"Mean latency    : "
        f"{overall['mean_latency_ms']:.2f} ms"
    )

    print()
    print("BY CATEGORY")
    print("-" * 72)

    for category, metrics in by_category.items():
        print(category)
        print(
            f"  Recall@{args.top_k}  : "
            f"{metrics['recall_at_k']:.4f}"
        )
        print(
            f"  MRR             : "
            f"{metrics['mrr']:.4f}"
        )
        print(
            f"  Mean latency    : "
            f"{metrics['mean_latency_ms']:.2f} ms"
        )

    print()
    print("CASES")
    print("-" * 72)

    for result in results:
        print(result["case_id"])
        print(f"  Query      : {result['query']}")
        print(
            f"  Expected   : "
            f"{result['expected_titles']}"
        )
        print(
            f"  Retrieved  : "
            f"{result['retrieved_titles']}"
        )
        print(
            f"  Matched    : "
            f"{result['matched_expected_titles']}"
        )
        print(
            f"  Recall     : "
            f"{result['recall_at_k']:.4f}"
        )
        print(
            f"  First rank : "
            f"{result['first_relevant_rank']}"
        )
        print(
            f"  MRR        : "
            f"{result['mrr']:.4f}"
        )
        print(
            f"  Latency    : "
            f"{result['latency_ms']:.2f} ms"
        )
        print()

    output = {
        "evaluation": {
            "name": "phase3_2_pageindex_qwen_navigation",
            "navigator": "QwenPageIndexNavigator",
            "model": args.model,
            "index_path": str(args.index_path),
            "cases_path": str(args.cases_path),
            "top_k": args.top_k,
            "toc_source": index.get("toc_source"),
        },
        "overall": overall,
        "by_category": by_category,
        "results": results,
    }

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    print(
        f"Saved results: {args.output_path}"
    )


if __name__ == "__main__":
    main()