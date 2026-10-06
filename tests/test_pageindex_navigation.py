from src.retrieval.pageindex import PageIndexNavigator


def sample_structure():
    return {
        "structure": [
            {
                "title": "I. Introduction",
                "node_id": "0001",
                "start_index": 1,
                "end_index": 5,
                "nodes": [],
            },
            {
                "title": "II. An Overview of the EV Consumer Market",
                "node_id": "0005",
                "start_index": 8,
                "end_index": 27,
                "nodes": [
                    {
                        "title": "1. Global adoption",
                        "node_id": "0006",
                        "start_index": 8,
                        "end_index": 13,
                        "nodes": [],
                    },
                    {
                        "title": "2. Regional trends",
                        "node_id": "0008",
                        "start_index": 13,
                        "end_index": 23,
                        "nodes": [],
                    },
                    {
                        "title": "3. Leading brands and models",
                        "node_id": "0009",
                        "start_index": 23,
                        "end_index": 25,
                        "nodes": [],
                    },
                ],
            },
        ]
    }


def test_navigation_finds_exact_structural_title():
    navigator = PageIndexNavigator(
        sample_structure()
    )

    results = navigator.navigate(
        "Regional trends",
        top_k=3,
    )

    assert results

    assert results[0].node["node_id"] == "0008"

    assert results[0].node["title"] == (
        "2. Regional trends"
    )


def test_navigation_preserves_hierarchy_path():
    navigator = PageIndexNavigator(
        sample_structure()
    )

    results = navigator.navigate(
        "Regional trends",
        top_k=1,
    )

    assert len(results) == 1

    assert results[0].path == (
        "II. An Overview of the EV Consumer Market",
        "2. Regional trends",
    )


def test_navigation_prefers_specific_child_over_broad_parent():
    navigator = PageIndexNavigator(
        sample_structure()
    )

    results = navigator.navigate(
        "EV Consumer Market Regional trends",
        top_k=2,
    )

    assert results

    node_ids = [
        result.node["node_id"]
        for result in results
    ]

    assert "0008" in node_ids


def test_navigation_is_deterministic():
    navigator = PageIndexNavigator(
        sample_structure()
    )

    first = navigator.navigate(
        "global adoption",
        top_k=3,
    )

    second = navigator.navigate(
        "global adoption",
        top_k=3,
    )

    assert [
        (
            item.node["node_id"],
            item.score,
            item.path,
        )
        for item in first
    ] == [
        (
            item.node["node_id"],
            item.score,
            item.path,
        )
        for item in second
    ]


def test_navigation_rejects_empty_query():
    navigator = PageIndexNavigator(
        sample_structure()
    )

    try:
        navigator.navigate("")
    except ValueError as exc:
        assert "query must not be empty" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for empty query"
        )


def test_navigation_rejects_invalid_top_k():
    navigator = PageIndexNavigator(
        sample_structure()
    )

    try:
        navigator.navigate(
            "regional trends",
            top_k=0,
        )
    except ValueError as exc:
        assert "top_k must be >= 1" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for invalid top_k"
        )