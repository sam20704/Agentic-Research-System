from src.document.models import (
    BoundingBox,
    Document,
    Element,
    Page,
)
from src.retrieval.chunking import (
    ChunkingConfig,
    chunk_document,
)


def make_document() -> Document:
    element = Element(
        element_id="element-001",
        element_type="paragraph",
        text=(
            "India has introduced semiconductor incentives to "
            "support domestic semiconductor manufacturing and "
            "strengthen the electronics supply chain."
        ),
        bbox=BoundingBox(
            x0=10,
            y0=20,
            x1=500,
            y1=100,
        ),
    )

    page = Page(
        page_number=1,
        text=element.text,
        width=600,
        height=800,
        elements=[element],
    )

    return Document(
        document_id="doc-001",
        source="test.pdf",
        source_path="test.pdf",
        file_hash="abc123",
        pages=[page],
        metadata={"parser": "pymupdf"},
    )


def test_document_is_chunked():
    document = make_document()

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-001"
    assert chunks[0].page_numbers == (1,)
    assert chunks[0].source == "test.pdf"


def test_chunk_preserves_element_provenance():
    document = make_document()

    chunks = chunk_document(document)

    assert chunks[0].element_ids == ("element-001",)
    assert len(chunks[0].bounding_boxes) == 1
    assert chunks[0].bounding_boxes[0].x0 == 10


def test_chunk_ids_are_deterministic():
    document = make_document()

    first = chunk_document(document)
    second = chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_large_text_is_split():
    document = make_document()

    config = ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_chars=20,
    )

    chunks = chunk_document(
        document,
        config=config,
    )

    assert len(chunks) > 1
    assert all(chunk.page_numbers == (1,) for chunk in chunks)


def test_invalid_chunking_config():
    try:
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=100,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected invalid overlap configuration to fail"
        )