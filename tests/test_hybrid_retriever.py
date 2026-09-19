from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import DocumentChunk, RetrievalResult


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


def chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc",
        text="example",
        page_numbers=(2,),
        source="source.pdf",
    )


# ------------------------------------------------------------------
# Fake dependencies (Dependency Injection)
# ------------------------------------------------------------------


class FakeBM25:
    def __init__(self):
        self.last_query = None
        self.last_top_k = None

    def search(self, query, top_k):
        self.last_query = query
        self.last_top_k = top_k

        return [
            RetrievalResult(
                chunk=chunk("chunk1"),
                score=2.0,
                rank=1,
                retrieval_method="bm25",
            )
        ]


class FakeDenseRetriever:
    def __init__(self):
        self.last_query = None
        self.last_top_k = None

    def retrieve(self, query, top_k):
        self.last_query = query
        self.last_top_k = top_k

        return [
            RetrievalResult(
                chunk=chunk("chunk1"),
                score=0.95,
                rank=1,
                retrieval_method="dense",
            ),
            RetrievalResult(
                chunk=chunk("chunk2"),
                score=0.80,
                rank=2,
                retrieval_method="dense",
            ),
        ]


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def make_retriever():
    bm25 = FakeBM25()
    dense = FakeDenseRetriever()

    retriever = HybridRetriever(
        bm25_retriever=bm25,
        vectorstore=None,
        embedder=None,
        fusion=ReciprocalRankFusion(),
        dense_retriever=dense,
    )

    return retriever, bm25, dense


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_hybrid_returns_fused_results():
    retriever, _, _ = make_retriever()

    results = retriever.retrieve("semiconductor")

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk1"
    assert results[0].retrieval_method == "hybrid"


def test_metadata_contains_provenance():
    retriever, _, _ = make_retriever()

    result = retriever.retrieve("semiconductor")[0]

    assert result.chunk.metadata["bm25_rank"] == 1
    assert result.chunk.metadata["dense_rank"] == 1

    assert "bm25" in result.chunk.metadata["retrieval_sources"]
    assert "dense" in result.chunk.metadata["retrieval_sources"]


def test_hybrid_calls_both_retrievers():
    retriever, bm25, dense = make_retriever()

    retriever.retrieve(
        "india semiconductor policy",
        bm25_top_k=5,
        dense_top_k=7,
    )

    assert bm25.last_query == "india semiconductor policy"
    assert bm25.last_top_k == 5

    assert dense.last_query == "india semiconductor policy"
    assert dense.last_top_k == 7