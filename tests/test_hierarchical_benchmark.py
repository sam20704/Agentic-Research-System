
"""Tests for the Phase 3.3 hierarchical benchmark."""

import json

from src.evaluation.run_hierarchical_benchmark import (
    benchmark,
    ground_truth_ids,
    load_chunks,
    load_queries,
    recall_at_k,
    reciprocal_rank,
    query_text,
)
from src.document.models import BoundingBox
from src.retrieval.models import DocumentChunk


def make_chunk(
    chunk_id: str,
    text: str,
    page: int,
    section: str | None = None,
) -> DocumentChunk:
    """Create a minimal benchmark chunk."""
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="benchmark-doc",
        text=text,
        page_numbers=[page],
        source="benchmark.pdf",
        section=section,
        bounding_boxes=[],
        element_ids=[],
        metadata={},
    )


def test_recall_at_k():
    relevant = {"chunk-1", "chunk-2"}

    assert recall_at_k(
        ["chunk-1", "chunk-3"],
        relevant,
        1,
    ) == 0.5

    assert recall_at_k(
        ["chunk-1", "chunk-2"],
        relevant,
        2,
    ) == 1.0


def test_recall_at_k_with_empty_ground_truth():
    assert recall_at_k(
        ["chunk-1"],
        set(),
        5,
    ) == 0.0


def test_reciprocal_rank():
    relevant = {"chunk-2"}

    assert reciprocal_rank(
        ["chunk-1", "chunk-2"],
        relevant,
    ) == 0.5

    assert reciprocal_rank(
        ["chunk-2"],
        relevant,
    ) == 1.0


def test_reciprocal_rank_when_not_retrieved():
    assert reciprocal_rank(
        ["chunk-1", "chunk-3"],
        {"chunk-2"},
    ) == 0.0


def test_query_text_accepts_supported_keys():
    assert query_text({"query": "test query"}) == "test query"
    assert query_text({"question": "test question"}) == "test question"
    assert query_text({"text": "test text"}) == "test text"


def test_ground_truth_ids_accepts_string_and_dict_entries():
    item = {
        "relevant_chunk_ids": [
            "chunk-1",
            {"chunk_id": "chunk-2"},
        ]
    }

    assert ground_truth_ids(item) == {
        "chunk-1",
        "chunk-2",
    }


def test_load_chunks_and_queries(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    ground_truth_path = tmp_path / "ground_truth.json"

    chunks = [
        make_chunk(
            "chunk-1",
            "Semiconductor policy.",
            1,
            section="Policy",
        )
    ]

    corpus_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "page_numbers": chunk.page_numbers,
                    "source": chunk.source,
                    "section": chunk.section,
                    "bounding_boxes": chunk.bounding_boxes,
                    "element_ids": chunk.element_ids,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]
        ),
        encoding="utf-8",
    )

    ground_truth_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query": "semiconductor policy",
                        "relevant_chunk_ids": ["chunk-1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded_chunks = load_chunks(corpus_path)
    loaded_queries = load_queries(ground_truth_path)

    assert len(loaded_chunks) == 1
    assert loaded_chunks[0].chunk_id == "chunk-1"

    assert len(loaded_queries) == 1
    assert loaded_queries[0]["query"] == "semiconductor policy"


def test_benchmark_writes_expected_result_structure(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    ground_truth_path = tmp_path / "ground_truth.json"
    output_path = tmp_path / "results.json"

    chunks = [
        make_chunk(
            "chunk-1",
            "Semiconductor policy and supply chain.",
            1,
            section="Semiconductors",
        ),
        make_chunk(
            "chunk-2",
            "Electric vehicle manufacturing.",
            2,
            section="Electric Vehicles",
        ),
        make_chunk(
            "chunk-3",
            "Global shipping capacity.",
            3,
            section="Supply Chain",
        ),
    ]

    corpus_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "page_numbers": chunk.page_numbers,
                    "source": chunk.source,
                    "section": chunk.section,
                    "bounding_boxes": chunk.bounding_boxes,
                    "element_ids": chunk.element_ids,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]
        ),
        encoding="utf-8",
    )

    ground_truth_path.write_text(
        json.dumps(
            [
                {
                    "query": "semiconductor policy",
                    "relevant_chunk_ids": ["chunk-1"],
                },
                {
                    "query": "electric vehicle manufacturing",
                    "relevant_chunk_ids": ["chunk-2"],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = benchmark(
        corpus_path=corpus_path,
        ground_truth_path=ground_truth_path,
        output_path=output_path,
        top_k=2,
    )

    assert output_path.exists()

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved["phase"] == "3.3"
    assert saved["method"] == "hierarchical_retrieval"
    assert saved["corpus"]["chunks"] == 3
    assert saved["ground_truth"]["queries"] == 2

    metrics = result["metrics"]

    assert 0.0 <= metrics["recall_at_5"] <= 1.0
    assert 0.0 <= metrics["recall_at_10"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert metrics["mean_latency_ms"] >= 0.0
    assert metrics["median_latency_ms"] >= 0.0
    assert metrics["index_construction_ms"] >= 0.0
    assert metrics["average_section_nodes_scored"] >= 0.0
    assert metrics["average_page_nodes_scored"] >= 0.0
    assert metrics["average_chunk_nodes_scored"] >= 0.0
    assert 0.0 <= metrics["average_chunk_candidate_reduction_ratio"] <= 1.0
    assert result["runtime"]["model"] == "deterministic / none"
    assert result["runtime"]["device"] == "CPU"
    assert result["index"]["nodes"] > 0

    assert len(result["per_query"]) == 2

    for item in result["per_query"]:
        assert "query" in item
        assert "retrieved_chunk_ids" in item
        assert "recall_at_5" in item
        assert "recall_at_10" in item
        assert "reciprocal_rank" in item
        assert "latency_ms" in item

    assert (
        result["baseline_comparison"]["status"]
        == "not_run"
    )

def test_load_chunks_reconstructs_canonical_bounding_boxes(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps([{
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "text": "Evidence with geometry.",
        "page_numbers": [1],
        "source": "reference.pdf",
        "section": "Evidence",
        "bounding_boxes": [{"x0": 1, "y0": 2, "x1": 3, "y1": 4}],
        "element_ids": ["element-1"],
        "metadata": {"origin": "canonical"},
    }]), encoding="utf-8")
    chunk = load_chunks(corpus_path)[0]
    assert chunk.bounding_boxes == (BoundingBox(1.0, 2.0, 3.0, 4.0),)
    assert chunk.element_ids == ("element-1",)
    assert chunk.metadata["origin"] == "canonical"


def test_benchmark_rejects_ground_truth_ids_not_in_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    ground_truth_path = tmp_path / "gt.json"
    output_path = tmp_path / "out.json"
    chunk = make_chunk("known", "Known evidence.", 1, "Evidence")
    corpus_path.write_text(json.dumps([{
        "chunk_id": chunk.chunk_id, "document_id": chunk.document_id,
        "text": chunk.text, "page_numbers": list(chunk.page_numbers),
        "source": chunk.source, "section": chunk.section,
        "bounding_boxes": [], "element_ids": [], "metadata": {},
    }]), encoding="utf-8")
    ground_truth_path.write_text(json.dumps([{
        "query": "unknown evidence", "relevant_chunk_ids": ["missing-id"]
    }]), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="unknown chunk IDs"):
        benchmark(corpus_path, ground_truth_path, output_path, top_k=1)
