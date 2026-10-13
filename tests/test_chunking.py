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

def test_chunking_preserves_explicit_hierarchy_metadata():
    element = Element(
        element_id="heading-aware-element",
        element_type="paragraph",
        text="Semiconductor supply chain policy and manufacturing details are described here.",
        bbox=BoundingBox(x0=1, y0=2, x1=300, y1=80),
        metadata={
            "hierarchical_section_path": "Industry > Semiconductors",
            "hierarchical_section_id": "heading-17",
        },
    )
    page = Page(
        page_number=7,
        text=element.text,
        elements=[element],
        metadata={"section_path": "Fallback Page Section"},
    )
    document = Document(
        document_id="doc-structure",
        source="structure.pdf",
        source_path="structure.pdf",
        file_hash="hash",
        pages=[page],
        metadata={"parser": "test"},
    )
    chunks = chunk_document(document, ChunkingConfig(min_chunk_chars=10))
    assert len(chunks) == 1
    assert chunks[0].metadata["hierarchical_section_path"] == "Industry > Semiconductors"
    assert chunks[0].metadata["hierarchical_section_id"] == "heading-17"
    assert chunks[0].metadata["section_path"] == "Fallback Page Section"
