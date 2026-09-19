"""Integration test for Phase 2.4 hybrid retrieval (deterministic version)."""

import pytest
from qdrant_client import QdrantClient

from src.document.models import BoundingBox
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import DocumentChunk
from src.retrieval.sparse.bm25 import BM25Retriever
from src.retrieval.vectorstore import QdrantVectorStore


# ------------------------------------------------------------------
# Fake deterministic embedder
# ------------------------------------------------------------------

class FakeEmbedder:
    """Maps query text to deterministic vectors."""

    def embed_query(self, query: str):
        query = query.lower()

        if "semiconductor" in query:
            return [1.0, 0.0, 0.0]

        if "vehicle" in query or "ev" in query:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------

def make_chunk(
    chunk_id: str,
    text: str,
    page: int,
) -> DocumentChunk:
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
        metadata={"page": page},
    )


# ------------------------------------------------------------------
# Integration: BM25 + Fake Embedder + Qdrant + RRF
# ------------------------------------------------------------------

@pytest.mark.integration
def test_hybrid_retriever_end_to_end():
    client = QdrantClient(":memory:")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="hybrid_integration",
        vector_size=3,
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

    embeddings = [
        [1.0, 0.0, 0.0],  # semiconductor
        [0.0, 1.0, 0.0],  # EV
        [0.8, 0.2, 0.0],  # semiconductor-related
    ]

    vectorstore.upsert(chunks, embeddings)

    bm25 = BM25Retriever(chunks)

    retriever = HybridRetriever(
        bm25_retriever=bm25,
        vectorstore=vectorstore,
        embedder=FakeEmbedder(),
    )

    results = retriever.retrieve(
        "India semiconductor policy",
        bm25_top_k=2,
        dense_top_k=2,
        final_top_k=2,
    )

    assert len(results) == 2

    assert results[0].chunk.chunk_id == "chunk1"
    assert results[0].retrieval_method == "hybrid"

    metadata = results[0].chunk.metadata

    assert metadata["bm25_rank"] == 1
    assert metadata["dense_rank"] == 1
    assert metadata["retrieval_sources"] == ["bm25", "dense"]
    assert "rrf_score" in metadata

    assert results[0].chunk.document_id == "doc1"
    assert results[0].chunk.page_numbers == (1,)
    assert results[0].chunk.source == "policy.pdf"
    assert results[0].chunk.element_ids == ("chunk1-element",)