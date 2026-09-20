
"""Integration test for ColBERT late-interaction retrieval."""

import pytest

from src.document.models import BoundingBox
from src.retrieval.colbert import ColBERTConfig, ColBERTRetriever
from src.retrieval.models import DocumentChunk


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def make_chunk(chunk_id: str, text: str, page: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        page_numbers=(page,),
        source="policy.pdf",
        bounding_boxes=(
            BoundingBox(
                x0=10,
                y0=20,
                x1=30,
                y1=40,
            ),
        ),
        element_ids=(f"{chunk_id}-element",),
        metadata={"section": "Policy"},
    )


# ------------------------------------------------------------------
# Integration test
# ------------------------------------------------------------------

@pytest.mark.integration
def test_colbert_retriever_end_to_end():
    retriever = ColBERTRetriever(
        config=ColBERTConfig(
            device="cpu",
            batch_size=2,
            default_top_k=2,
        )
    )

    chunks = [
        make_chunk(
            "chunk1",
            "India semiconductor policy provides manufacturing incentives.",
            1,
        ),
        make_chunk(
            "chunk2",
            "FAME II supports electric vehicle adoption.",
            2,
        ),
        make_chunk(
            "chunk3",
            "Global semiconductor supply chain resilience report.",
            5,
        ),
    ]

    retriever.build_index(chunks)

    assert retriever.index_size() == 3

    results = retriever.retrieve(
        "India semiconductor policy",
        top_k=2,
    )

    # --------------------------------------------------------------
    # Ranking
    # --------------------------------------------------------------

    assert len(results) == 2
    assert results[0].retrieval_method == "colbert"

    # Semiconductor chunk should be retrieved.
    retrieved_ids = [r.chunk.chunk_id for r in results]

    assert "chunk1" in retrieved_ids

    # --------------------------------------------------------------
    # Provenance preservation
    # --------------------------------------------------------------

    result = next(r for r in results if r.chunk.chunk_id == "chunk1")

    assert result.chunk.document_id == "doc1"
    assert result.chunk.page_numbers == (1,)
    assert result.chunk.source == "policy.pdf"

    assert result.chunk.element_ids == ("chunk1-element",)

    assert result.chunk.metadata["section"] == "Policy"
    assert result.chunk.metadata["retrieval_sources"] == ["colbert"]
    assert "colbert_score" in result.chunk.metadata

    bbox = result.chunk.bounding_boxes[0]

    assert bbox.x0 == 10
    assert bbox.y0 == 20
    assert bbox.x1 == 30
    assert bbox.y1 == 40