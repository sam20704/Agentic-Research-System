"""Integration test for Phase 2.4 using the real BGE-M3 embedding model."""

import pytest
from qdrant_client import QdrantClient

from src.document.models import BoundingBox
from src.retrieval.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import DocumentChunk
from src.retrieval.sparse.bm25 import BM25Retriever
from src.retrieval.vectorstore import QdrantVectorStore


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


@pytest.mark.integration
def test_hybrid_retriever_end_to_end_bge_m3():
    embedder = SentenceTransformerEmbedding()

    client = QdrantClient(":memory:")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="hybrid_bge_m3",
        vector_size=embedder.dimension,
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

    embeddings = embedder.embed_chunks(chunks)

    vectorstore.upsert(chunks, embeddings)

    bm25 = BM25Retriever(chunks)

    retriever = HybridRetriever(
        bm25_retriever=bm25,
        vectorstore=vectorstore,
        embedder=embedder,
    )

    results = retriever.retrieve(
        "India semiconductor policy",
        bm25_top_k=3,
        dense_top_k=3,
        final_top_k=3,
    )

    assert len(results) >= 1

    top = results[0]

    assert top.chunk.chunk_id == "chunk1"
    assert top.retrieval_method == "hybrid"

    metadata = top.chunk.metadata

    assert metadata["bm25_rank"] == 1
    assert metadata["dense_rank"] == 1
    assert metadata["retrieval_sources"] == ["bm25", "dense"]
    assert "rrf_score" in metadata

    assert top.chunk.document_id == "doc1"
    assert top.chunk.page_numbers == (1,)
    assert top.chunk.source == "policy.pdf"