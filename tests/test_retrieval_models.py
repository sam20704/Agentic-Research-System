import pytest

from src.document.models import BoundingBox
from src.retrieval.models import DocumentChunk, RetrievalResult


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        text="India semiconductor policy provides incentives.",
        page_numbers=(3,),
        source="test.pdf",
        bounding_boxes=(
            BoundingBox(
                x0=10.0,
                y0=20.0,
                x1=300.0,
                y1=80.0,
            ),
        ),
        element_ids=("element-001",),
        metadata={"parser": "pymupdf"},
    )


def test_document_chunk_creation():
    chunk = make_chunk()

    assert chunk.chunk_id == "chunk-001"
    assert chunk.document_id == "doc-001"
    assert chunk.primary_page == 3
    assert chunk.page_count == 1
    assert chunk.element_ids == ("element-001",)


def test_document_chunk_supports_multiple_pages():
    chunk = DocumentChunk(
        chunk_id="chunk-002",
        document_id="doc-001",
        text="Content spanning two pages.",
        page_numbers=(3, 4),
        source="test.pdf",
    )

    assert chunk.primary_page == 3
    assert chunk.page_count == 2


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="text"):
        DocumentChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            text="   ",
            page_numbers=(1,),
            source="test.pdf",
        )


def test_invalid_page_number_is_rejected():
    with pytest.raises(ValueError, match="page numbers"):
        DocumentChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            text="Some text.",
            page_numbers=(0,),
            source="test.pdf",
        )


def test_provenance_lengths_must_match():
    with pytest.raises(
        ValueError,
        match="bounding_boxes and element_ids",
    ):
        DocumentChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            text="Some text.",
            page_numbers=(1,),
            source="test.pdf",
            bounding_boxes=(
                BoundingBox(
                    x0=0,
                    y0=0,
                    x1=100,
                    y1=100,
                ),
            ),
            element_ids=(),
        )


def test_retrieval_result():
    chunk = make_chunk()

    result = RetrievalResult(
        chunk=chunk,
        score=0.91,
        rank=1,
        retrieval_method="dense",
    )

    assert result.chunk.chunk_id == "chunk-001"
    assert result.score == 0.91
    assert result.rank == 1
    assert result.retrieval_method == "dense"


def test_invalid_rank_is_rejected():
    chunk = make_chunk()

    with pytest.raises(ValueError, match="rank"):
        RetrievalResult(
            chunk=chunk,
            score=0.5,
            rank=0,
            retrieval_method="dense",
        )