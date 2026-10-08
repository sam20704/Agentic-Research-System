from pathlib import Path

import pytest

from src.document import BoundingBox, Document, Element, Page
from src.retrieval.models import DocumentChunk, RetrievalResult
from src.retrieval.pageindex import PageIndexConfig, PageIndexRetriever


@pytest.fixture
def document(tmp_path: Path) -> Document:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"placeholder")

    return Document(
        document_id="doc-1",
        source="sample.pdf",
        source_path=str(pdf_path),
        file_hash="hash",
        pages=[
            Page(
                page_number=1,
                text="Introduction to semiconductor policy.",
                width=100,
                height=100,
                elements=[
                    Element(
                        element_id="e1",
                        element_type="text",
                        text="Introduction to semiconductor policy.",
                        bbox=BoundingBox(0, 0, 10, 10),
                    )
                ],
            )
        ],
        metadata={},
    )


def test_pageindex_config_does_not_require_llm_model():
    config = PageIndexConfig()

    assert config.index_path is None


def test_pageindex_maps_structure_to_existing_retrieval_contract(
    monkeypatch,
    document,
):
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id=document.document_id,
            text=document.pages[0].text,
            page_numbers=(1,),
            source=document.source,
            section="Semiconductor Policy",
            bounding_boxes=(
                BoundingBox(0, 0, 10, 10),
            ),
            element_ids=("e1",),
            metadata={
                "fixture": "provenance",
            },
        )
    ]

    monkeypatch.setattr(
        PageIndexRetriever,
        "_build_structure",
        lambda self: {
            "doc_name": "sample.pdf",
            "structure": [
                {
                    "title": "Semiconductor Policy",
                    "node_id": "0001",
                    "start_index": 1,
                    "end_index": 1,
                    "nodes": [],
                }
            ],
        },
    )

    results = PageIndexRetriever(
        document,
        chunks,
    ).retrieve(
        "semiconductor policy",
        top_k=1,
    )

    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)

    # Retrieval contract.
    assert results[0].retrieval_method == "pageindex"
    assert results[0].chunk.chunk_id == chunks[0].chunk_id

    # Canonical document/chunk provenance.
    assert results[0].chunk.document_id == document.document_id
    assert results[0].chunk.page_numbers == (1,)
    assert results[0].chunk.source == document.source
    assert results[0].chunk.section == "Semiconductor Policy"
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].chunk.element_ids == ("e1",)
    assert results[0].chunk.bounding_boxes == (
        BoundingBox(0, 0, 10, 10),
    )

    # Existing chunk metadata must be preserved.
    assert results[0].chunk.metadata["fixture"] == "provenance"

    # PageIndex-specific provenance.
    assert results[0].chunk.metadata["retrieval_sources"] == [
        "pageindex"
    ]
    assert results[0].chunk.metadata["pageindex_node_id"] == "0001"
    assert (
        results[0].chunk.metadata["pageindex_title"]
        == "Semiconductor Policy"
    )
    assert results[0].chunk.metadata["pageindex_start_index"] == 1
    assert results[0].chunk.metadata["pageindex_end_index"] == 1


def test_pageindex_loads_persisted_index(tmp_path, document):
    index_path = tmp_path / "sample.json"

    index_path.write_text(
        """
        {
          "doc_name": "sample.pdf",
          "toc_source": "bookmarks",
          "structure": [
            {
              "title": "Semiconductor Policy",
              "node_id": "0001",
              "start_index": 1,
              "end_index": 1,
              "nodes": []
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    retriever = PageIndexRetriever(
        document,
        [],
        config=PageIndexConfig(index_path=index_path),
    )

    assert retriever.structure["toc_source"] == "bookmarks"
    assert len(retriever.structure["structure"]) == 1
    assert retriever.structure["structure"][0]["node_id"] == "0001"


def test_pageindex_missing_index_fails_clearly(tmp_path, document):
    missing_index = tmp_path / "does-not-exist.json"

    with pytest.raises(
        FileNotFoundError,
        match="PageIndex index not found",
    ):
        PageIndexRetriever(
            document,
            [],
            config=PageIndexConfig(index_path=missing_index),
        )


def test_pageindex_rejects_invalid_structure(tmp_path, document):
    index_path = tmp_path / "invalid.json"

    index_path.write_text(
        """
        {
          "doc_name": "sample.pdf",
          "structure": [
            {
              "title": "Broken Node",
              "node_id": "0001",
              "start_index": 1
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing required field 'end_index'",
    ):
        PageIndexRetriever(
            document,
            [],
            config=PageIndexConfig(index_path=index_path),
        )
