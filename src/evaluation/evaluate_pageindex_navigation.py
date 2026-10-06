"""Phase 3.2 evaluation for deterministic PageIndex navigation.

This evaluation is intentionally separate from the frozen Phase 3.1
retrieval benchmark.

It evaluates the current PageIndexNavigator directly against the persisted
PageIndex Flash tree and measures structural-node retrieval quality.

No embeddings, vector database, BM25, ColBERT, or LLM calls are used here.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.retrieval.pageindex import PageIndexNavigator


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
    / "phase3_2_pageindex_navigation_results.json"
)

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class NavigationCase:
    """One Phase 3.2 navigation evaluation case."""

    case_id: str
    category: str
    query: str
    expected_titles: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "NavigationCase":
        required = (
            "case_id",
            "category",
            "query",
            "expected_titles",
        )

        missing = [
            field
            for field in required
            if field not in data
        ]

        if missing:
            raise ValueError(
                "Navigation case is missing required fields: "
                + ", ".join(missing)
            )

        expected_titles = data["expected_titles"]

        if not isinstance(expected_titles, list):
            raise ValueError(
                f"expected_titles must be a list for case "
                f"{data['case_id']!r}"
            )

        if not expected_titles:
            raise ValueError(
                f"expected_titles must not be empty for case "
                f"{data['case_id']!r}"
            )

        query = str(data["query"]).strip()

        if not query:
            raise ValueError(
                f"query must not be empty for case "
                f"{data['case_id']!r}"
            )

        return cls(
            case_id=str(data["case_id"]),
            category=str(data["category"]),
            query=query,
            expected_titles=tuple(
                str(title).strip()
                for title in expected_titles
                if str(title).strip()
            ),
        )


@dataclass(frozen=True)
class NavigationEvaluationResult:
    """Per-case evaluation result."""

    case_id: str
    category: str
    query: str
    expected_titles: tuple[str, ...]
    top_k: int
    latency_ms: float
    retrieved_titles: tuple[str, ...]
    retrieved_node_ids: tuple[str, ...]
    retrieved_paths: tuple[tuple[str, ...], ...]
    matched_expected_titles: tuple[str, ...]
    recall_at_k: float
    first_relevant_rank: int | None
    mrr: float


def load_cases(
    path: str | Path,
) -> list[NavigationCase]:
    """Load Phase 3.2 navigation cases from JSON."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Navigation cases not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            "Navigation evaluation cases must be a JSON list."
        )

    cases = [
        NavigationCase.from_dict(item)
        for item in data
    ]

    if not cases:
        raise ValueError(
            "Navigation evaluation cases must not be empty."
        )

    return cases


def load_index(
    path: str | Path,
) -> dict[str, Any]:
    """Load and minimally validate a persisted PageIndex tree."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"PageIndex index not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "PageIndex index must contain a JSON object."
        )

    structure = data.get("structure")

    if not isinstance(structure, list):
        raise ValueError(
            "PageIndex index must contain a 'structure' list."
        )

    if not structure:
        raise ValueError(
            "PageIndex index structure must not be empty."
        )

    return data


def _normalize_title(
    title: str,
) -> str:
    """Normalize titles for evaluation matching."""

    return " ".join(
        str(title)
        .strip()
        .lower()
        .split()
    )


def _matched_expected_titles(
    expected_titles: tuple[str, ...],
    retrieved_titles: tuple[str, ...],
) -> tuple[str, ...]:
    """Return expected titles appearing in the retrieved set."""

    retrieved_normalized = {
        _normalize_title(title)
        for title in retrieved_titles
    }

    matched = []

    for expected in expected_titles:
        if (
            _normalize_title(expected)
            in retrieved_normalized
        ):
            matched.append(expected)

    return tuple(matched)


def _first_relevant_rank(
    expected_titles: tuple[str, ...],
    retrieved_titles: tuple[str, ...],
) -> int | None:
    """Return the first rank containing an expected title."""

    expected_normalized = {
        _normalize_title(title)
        for title in expected_titles
    }

    for rank, title in enumerate(
        retrieved_titles,
        start=1,
    ):
        if _normalize_title(title) in expected_normalized:
            return rank

    return None


def evaluate_case(
    navigator: PageIndexNavigator,
    case: NavigationCase,
    top_k: int,
) -> NavigationEvaluationResult:
    """Evaluate one navigation case."""

    started = time.perf_counter()

    candidates = navigator.navigate(
        case.query,
        top_k=top_k,
    )

    latency_ms = (
        time.perf_counter() - started
    ) * 1000.0

    retrieved_titles = tuple(
        str(candidate.node.get("title", ""))
        for candidate in candidates
    )

    retrieved_node_ids = tuple(
        str(candidate.node.get("node_id", ""))
        for candidate in candidates
    )

    retrieved_paths = tuple(
        tuple(candidate.path)
        for candidate in candidates
    )

    matched = _matched_expected_titles(
        case.expected_titles,
        retrieved_titles,
    )

    recall_at_k = (
        len(matched)
        / len(case.expected_titles)
    )

    first_rank = _first_relevant_rank(
        case.expected_titles,
        retrieved_titles,
    )

    mrr = (
        1.0 / first_rank
        if first_rank is not None
        else 0.0
    )

    return NavigationEvaluationResult(
        case_id=case.case_id,
        category=case.category,
        query=case.query,
        expected_titles=case.expected_titles,
        top_k=top_k,
        latency_ms=latency_ms,
        retrieved_titles=retrieved_titles,
        retrieved_node_ids=retrieved_node_ids,
        retrieved_paths=retrieved_paths,
        matched_expected_titles=matched,
        recall_at_k=recall_at_k,
        first_relevant_rank=first_rank,
        mrr=mrr,
    )


def _aggregate(
    results: list[NavigationEvaluationResult],
) -> dict[str, Any]:
    """Build aggregate evaluation statistics."""

    if not results:
        return {
            "cases": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "mean_latency_ms": 0.0,
        }

    return {
        "cases": len(results),
        "recall_at_k": sum(
            result.recall_at_k
            for result in results
        ) / len(results),
        "mrr": sum(
            result.mrr
            for result in results
        ) / len(results),
        "mean_latency_ms": sum(
            result.latency_ms
            for result in results
        ) / len(results),
    }


def evaluate(
    index_path: str | Path,
    cases_path: str | Path,
    top_k: int,
) -> dict[str, Any]:
    """Run the complete deterministic navigation evaluation."""

    if top_k < 1:
        raise ValueError(
            "top_k must be >= 1"
        )

    index = load_index(index_path)
    cases = load_cases(cases_path)

    navigator = PageIndexNavigator(index)

    results = [
        evaluate_case(
            navigator=navigator,
            case=case,
            top_k=top_k,
        )
        for case in cases
    ]

    by_category: dict[str, list[NavigationEvaluationResult]] = {}

    for result in results:
        by_category.setdefault(
            result.category,
            [],
        ).append(result)

    category_summaries = {
        category: _aggregate(category_results)
        for category, category_results
        in sorted(by_category.items())
    }

    return {
        "evaluation": {
            "name": "phase3_2_pageindex_navigation",
            "navigator": "PageIndexNavigator",
            "index_path": str(
                Path(index_path)
            ),
            "cases_path": str(
                Path(cases_path)
            ),
            "top_k": top_k,
            "toc_source": index.get(
                "toc_source"
            ),
        },
        "overall": _aggregate(results),
        "by_category": category_summaries,
        "results": [
            asdict(result)
            for result in results
        ],
    }


def save_results(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    """Persist evaluation results."""

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
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")


def run() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic PageIndex "
            "structural navigation."
        )
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    args = parser.parse_args()

    payload = evaluate(
        index_path=args.index,
        cases_path=args.cases,
        top_k=args.top_k,
    )

    save_results(
        args.output,
        payload,
    )

    evaluation = payload["evaluation"]
    overall = payload["overall"]

    print("=" * 72)
    print(
        "PHASE 3.2 — PAGEINDEX NAVIGATION EVALUATION"
    )
    print("=" * 72)
    print(
        f"Navigator       : "
        f"{evaluation['navigator']}"
    )
    print(
        f"TOC source      : "
        f"{evaluation['toc_source']}"
    )
    print(
        f"Top-k           : "
        f"{evaluation['top_k']}"
    )
    print(
        f"Cases           : "
        f"{overall['cases']}"
    )
    print(
        f"Recall@{evaluation['top_k']}      : "
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

    for category, summary in payload[
        "by_category"
    ].items():
        print(category)
        print(
            f"  Cases         : "
            f"{summary['cases']}"
        )
        print(
            f"  Recall@{evaluation['top_k']}     : "
            f"{summary['recall_at_k']:.4f}"
        )
        print(
            f"  MRR           : "
            f"{summary['mrr']:.4f}"
        )
        print(
            f"  Mean latency  : "
            f"{summary['mean_latency_ms']:.2f} ms"
        )
        print()

    print("CASES")
    print("-" * 72)

    for result in payload["results"]:
        print(
            f"{result['case_id']}"
        )
        print(
            f"  Query          : "
            f"{result['query']}"
        )
        print(
            f"  Expected       : "
            f"{list(result['expected_titles'])}"
        )
        print(
            f"  Retrieved      : "
            f"{list(result['retrieved_titles'])}"
        )
        print(
            f"  Matched        : "
            f"{list(result['matched_expected_titles'])}"
        )
        print(
            f"  Recall         : "
            f"{result['recall_at_k']:.4f}"
        )
        print(
            f"  First rank     : "
            f"{result['first_relevant_rank']}"
        )
        print(
            f"  MRR            : "
            f"{result['mrr']:.4f}"
        )
        print(
            f"  Latency        : "
            f"{result['latency_ms']:.2f} ms"
        )
        print()

    print(
        f"Saved results: {args.output}"
    )


if __name__ == "__main__":
    run()