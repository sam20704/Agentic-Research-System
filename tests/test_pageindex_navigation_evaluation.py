import json
from pathlib import Path

from src.evaluation.evaluate_pageindex_navigation import (
    DEFAULT_CASES_PATH,
    DEFAULT_INDEX_PATH,
    evaluate,
    load_cases,
    load_index,
)


def test_phase3_2_navigation_cases_are_valid():
    cases = load_cases(DEFAULT_CASES_PATH)

    assert len(cases) == 3

    categories = {
        case.category
        for case in cases
    }

    assert categories == {
        "exact_structural",
        "semantic_structural",
        "multihop_complex",
    }

    assert all(
        case.query.strip()
        for case in cases
    )

    assert all(
        case.expected_titles
        for case in cases
    )


def test_phase3_2_flash_index_is_valid():
    index = load_index(DEFAULT_INDEX_PATH)

    assert index["toc_source"] == "bookmarks"

    structure = index["structure"]

    assert len(structure) == 13

    def walk(nodes):
        for node in nodes:
            yield node

            children = node.get("nodes") or []

            if isinstance(children, list):
                yield from walk(children)

    nodes = list(walk(structure))

    assert len(nodes) == 82

    assert all(
        "node_id" in node
        for node in nodes
    )

    assert all(
        "title" in node
        for node in nodes
    )

    assert all(
        "start_index" in node
        and "end_index" in node
        for node in nodes
    )


def test_phase3_2_navigation_evaluation_runs():
    payload = evaluate(
        index_path=DEFAULT_INDEX_PATH,
        cases_path=DEFAULT_CASES_PATH,
        top_k=5,
    )

    assert payload["evaluation"][
        "navigator"
    ] == "PageIndexNavigator"

    assert payload["evaluation"][
        "toc_source"
    ] == "bookmarks"

    assert payload["overall"]["cases"] == 3

    assert len(
        payload["results"]
    ) == 3

    for result in payload["results"]:
        assert result["retrieved_titles"]

        assert (
            result["latency_ms"] >= 0.0
        )

        assert 0.0 <= (
            result["recall_at_k"]
        ) <= 1.0

        assert 0.0 <= (
            result["mrr"]
        ) <= 1.0


def test_exact_structural_case_retrieves_regional_trends():
    payload = evaluate(
        index_path=DEFAULT_INDEX_PATH,
        cases_path=DEFAULT_CASES_PATH,
        top_k=5,
    )

    result = next(
        item
        for item in payload["results"]
        if item["case_id"]
        == "exact_structural_regional_trends"
    )

    assert (
        "2. Regional trends"
        in result["matched_expected_titles"]
    )


def test_evaluation_output_is_json_serializable():
    payload = evaluate(
        index_path=DEFAULT_INDEX_PATH,
        cases_path=DEFAULT_CASES_PATH,
        top_k=5,
    )

    json.dumps(
        payload,
        ensure_ascii=False,
    )