from pathlib import Path

from src.document import BoundingBox, Document, Element, Page
from src.retrieval.models import DocumentChunk
from src.retrieval.pageindex import PageIndexRetriever


def test_pageindex_path_is_independent_of_vector_components(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"placeholder")
    document = Document(
        document_id="doc-iso",
        source="sample.pdf",
        source_path=str(pdf_path),
        file_hash="hash",
        pages=[
            Page(
                page_number=1,
                text="Semiconductor fabrication policy.",
                width=100,
                height=100,
                elements=[
                    Element(
                        element_id="e1",
                        element_type="text",
                        text="Semiconductor fabrication policy.",
                        bbox=BoundingBox(0, 0, 10, 10),
                    )
                ],
            )
        ],
    )
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            document_id=document.document_id,
            text=document.pages[0].text,
            page_numbers=(1,),
            source=document.source,
        )
    ]

    monkeypatch.setenv("PAGEINDEX_MODEL", "ollama/test-model")
    monkeypatch.setattr(
        PageIndexRetriever,
        "_build_structure",
        lambda self: {
            "doc_name": "sample.pdf",
            "structure": [
                {
                    "title": "Semiconductor Fabrication",
                    "node_id": "0001",
                    "start_index": 1,
                    "end_index": 1,
                    "nodes": [],
                }
            ],
        },
    )

    results = PageIndexRetriever(document, chunks).retrieve(
        "semiconductor fabrication"
    )

    assert results
    assert all(result.retrieval_method == "pageindex" for result in results)
    assert all(
        result.chunk.metadata["retrieval_sources"] == ["pageindex"]
        for result in results
    )
