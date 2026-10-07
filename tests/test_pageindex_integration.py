import json
from pathlib import Path

from src.document.models import BoundingBox, Document
from src.retrieval.models import DocumentChunk
from src.retrieval.pageindex import PageIndexConfig, PageIndexRetriever


REPO_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    REPO_ROOT
    / "data"
    / "references"
    / "Global_Supply_Chain_Report_DEC2025_EV_1.pdf"
)

PAGEINDEX_PATH = (
    REPO_ROOT
    / ".pageindex"
    / "Global_Supply_Chain_Report_DEC2025_EV_1.json"
)

CORPUS_PATH = (
    REPO_ROOT
    / "data"
    / "benchmarks"
    / "retrieval_corpus.json"
)

DOCUMENT_ID = "doc_3fdad50b4105bb6e"
SOURCE = "Global_Supply_Chain_Report_DEC2025_EV_1.pdf"


def _load_real_document_chunks() -> list[DocumentChunk]:
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        corpus = json.load(handle)

    entries = [
        entry
        for entry in corpus
        if entry.get("document_id") == DOCUMENT_ID
    ]

    assert entries, (
        f"No retrieval-corpus entries found for document "
        f"{DOCUMENT_ID!r}"
    )

    chunks: list[DocumentChunk] = []

    for entry in entries:
        bounding_boxes = tuple(
            BoundingBox(
                x0=float(box["x0"]),
                y0=float(box["y0"]),
                x1=float(box["x1"]),
                y1=float(box["y1"]),
            )
            for box in entry.get("bounding_boxes", [])
        )

        chunks.append(
            DocumentChunk(
                chunk_id=entry["chunk_id"],
                document_id=entry["document_id"],
                text=entry["text"],
                page_numbers=tuple(entry["page_numbers"]),
                source=entry["source"],
                section=entry.get("section"),
                bounding_boxes=bounding_boxes,
                element_ids=tuple(entry.get("element_ids", [])),
                metadata=dict(entry.get("metadata", {})),
            )
        )

    return chunks


def _flatten_nodes(nodes: list[dict]) -> list[dict]:
    flattened: list[dict] = []

    for node in nodes:
        flattened.append(node)
        flattened.extend(_flatten_nodes(node.get("nodes") or []))

    return flattened


def test_pageindex_real_long_document_integration():
    """Exercise PageIndex retrieval against the real 82-node document tree."""

    assert PDF_PATH.is_file(), f"Real PDF not found: {PDF_PATH}"
    assert PAGEINDEX_PATH.is_file(), (
        f"Persisted PageIndex artifact not found: {PAGEINDEX_PATH}"
    )
    assert CORPUS_PATH.is_file(), (
        f"Retrieval corpus not found: {CORPUS_PATH}"
    )

    with PAGEINDEX_PATH.open("r", encoding="utf-8") as handle:
        pageindex = json.load(handle)

    structure = pageindex["structure"]

    assert isinstance(structure, list)
    assert structure

    all_nodes = _flatten_nodes(structure)

    # The Phase 3.2 benchmark artifact is the real 82-node tree.
    assert len(all_nodes) == 82

    titles = {str(node.get("title")) for node in all_nodes}

    assert "2. Regional trends" in titles
    assert "2.5 Supply chain constraints" in titles

    chunks = _load_real_document_chunks()

    document = Document(
        document_id=DOCUMENT_ID,
        source=SOURCE,
        source_path=str(PDF_PATH),
        file_hash="integration-test-real-document",
    )

    retriever = PageIndexRetriever(
        document=document,
        chunks=chunks,
        config=PageIndexConfig(index_path=PAGEINDEX_PATH),
    )

    results = retriever.retrieve(
        "2. Regional trends",
        top_k=5,
    )

    assert results

    assert all(
        result.retrieval_method == "pageindex"
        for result in results
    )

    assert all(
        result.chunk.document_id == DOCUMENT_ID
        for result in results
    )

    assert all(
        result.chunk.source == SOURCE
        for result in results
    )

    assert all(
        result.chunk.metadata["retrieval_sources"] == ["pageindex"]
        for result in results
    )

    # The deterministic navigator should retrieve the exact
    # structural node from the real PageIndex hierarchy.
    regional_results = [
        result
        for result in results
        if result.chunk.metadata.get("pageindex_title")
        == "2. Regional trends"
    ]

    assert regional_results, (
        "PageIndex did not retrieve the expected "
        "'2. Regional trends' node"
    )

    for result in regional_results:
        node_start = int(
            result.chunk.metadata["pageindex_start_index"]
        )
        node_end = int(
            result.chunk.metadata["pageindex_end_index"]
        )

        assert all(
            node_start <= page <= node_end
            for page in result.chunk.page_numbers
        )

        assert result.chunk.metadata["pageindex_node_id"]
        assert result.chunk.metadata["pageindex_node_path"]

        assert result.chunk.element_ids
        assert result.chunk.bounding_boxes
