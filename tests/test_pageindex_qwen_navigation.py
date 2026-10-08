import pytest

from src.retrieval.pageindex.qwen_navigator import (
    QwenPageIndexNavigator,
)


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
                ],
            },
        ]
    }


class FakeResponse:
    def __init__(
        self,
        content,
        status_code=200,
    ):
        self.status_code = status_code
        self._content = content
        self.text = content

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(
                f"HTTP {self.status_code}"
            )

    def json(self):
        return {
            "message": {
                "role": "assistant",
                "content": self._content,
            }
        }


def make_navigator(monkeypatch, content):
    def fake_post(*args, **kwargs):
        return FakeResponse(content)

    monkeypatch.setattr(
        "src.retrieval.pageindex.qwen_navigator.requests.post",
        fake_post,
    )

    return QwenPageIndexNavigator(
        sample_structure(),
        model="qwen3.5:4b",
    )


def test_qwen_navigation_accepts_existing_node_ids(
    monkeypatch,
):
    navigator = make_navigator(
        monkeypatch,
        '{"node_ids": ["0008", "0006"]}',
    )

    results = navigator.navigate(
        "regional EV trends",
        top_k=2,
    )

    assert [item.node["node_id"] for item in results] == [
        "0008",
        "0006",
    ]


def test_qwen_navigation_preserves_paths(monkeypatch):
    navigator = make_navigator(
        monkeypatch,
        '{"node_ids": ["0008"]}',
    )

    results = navigator.navigate(
        "regional trends",
        top_k=1,
    )

    assert results[0].path == (
        "II. An Overview of the EV Consumer Market",
        "2. Regional trends",
    )


def test_qwen_navigation_rejects_unknown_node_id(
    monkeypatch,
):
    navigator = make_navigator(
        monkeypatch,
        '{"node_ids": ["9999"]}',
    )

    with pytest.raises(
        ValueError,
        match="unknown node_id",
    ):
        navigator.navigate(
            "regional trends",
            top_k=1,
        )


def test_qwen_navigation_rejects_empty_query(
    monkeypatch,
):
    navigator = make_navigator(
        monkeypatch,
        '{"node_ids": ["0008"]}',
    )

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        navigator.navigate("")


def test_qwen_navigation_rejects_invalid_top_k(
    monkeypatch,
):
    navigator = make_navigator(
        monkeypatch,
        '{"node_ids": ["0008"]}',
    )

    with pytest.raises(
        ValueError,
        match="top_k must be >= 1",
    ):
        navigator.navigate(
            "regional trends",
            top_k=0,
        )


def test_qwen_navigation_rejects_malformed_response(
    monkeypatch,
):
    navigator = make_navigator(
        monkeypatch,
        "not json",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        navigator.navigate(
            "regional trends",
            top_k=1,
        )