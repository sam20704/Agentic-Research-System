"""Integration tests for the Qdrant vector store."""

import pytest
from qdrant_client import QdrantClient

from src.document.models import BoundingBox
from src.retrieval.models import DocumentChunk
from src.retrieval.vectorstore import QdrantVectorStore


def make_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    page: int,
    section: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
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
        metadata={"section": section},
    )


@pytest.mark.integration
def test_qdrant_dense_search_returns_ranked_results():
    """Dense retrieval should return cosine-ranked results with provenance."""

    client = QdrantClient(":memory:")

    store = QdrantVectorStore(
        client=client,
        collection_name="integration_test_collection",
        vector_size=3,
    )

    chunk_a = make_chunk(
        chunk_id="chunkA",
        document_id="doc1",
        text="Semiconductor manufacturing incentives",
        page=1,
        section="Semiconductors",
    )

    chunk_b = make_chunk(
        chunk_id="chunkB",
        document_id="doc1",
        text="Electric vehicle adoption incentives",
        page=2,
        section="EV",
    )

    chunk_c = make_chunk(
        chunk_id="chunkC",
        document_id="doc2",
        text="Global semiconductor supply chains",
        page=5,
        section="Supply Chain",
    )

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    inserted = store.upsert(
        [chunk_a, chunk_b, chunk_c],
        embeddings,
    )

    assert inserted == 3
    assert store.count() == 3

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].chunk.chunk_id == "chunkA"
    assert results[0].retrieval_method == "dense"

    assert results[0].score >= results[1].score
    assert results[1].score >= results[2].score

    retrieved = results[0].chunk

    assert retrieved.document_id == "doc1"
    assert retrieved.page_numbers == (1,)
    assert retrieved.source == "policy.pdf"
    assert retrieved.element_ids == ("chunkA-element",)
    assert retrieved.metadata["section"] == "Semiconductors"

    bbox = retrieved.bounding_boxes[0]

    assert bbox.x0 == 10
    assert bbox.y0 == 20
    assert bbox.x1 == 30
    assert bbox.y1 == 40