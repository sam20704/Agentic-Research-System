
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.models import DocumentChunk, RetrievalResult


def chunk(chunk_id):

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc",
        text="example",
        page_numbers=(2,),
        source="source.pdf",
    )


class FakeBM25:

    def search(self, query, top_k):

        return [
            RetrievalResult(
                chunk=chunk("chunk1"),
                score=2,
                rank=1,
                retrieval_method="bm25",
            )
        ]


class FakeDense:

    def retrieve(self, query, top_k):

        return [
            RetrievalResult(
                chunk=chunk("chunk1"),
                score=.95,
                rank=1,
                retrieval_method="dense",
            ),
            RetrievalResult(
                chunk=chunk("chunk2"),
                score=.80,
                rank=2,
                retrieval_method="dense",
            ),
        ]


def test_hybrid_returns_fused_results():

    retriever = HybridRetriever.__new__(HybridRetriever)

    retriever.bm25 = FakeBM25()
    retriever.dense = FakeDense()
    retriever.fusion = ReciprocalRankFusion()

    results = retriever.retrieve("semiconductor")

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk1"


def test_metadata_contains_provenance():

    retriever = HybridRetriever.__new__(HybridRetriever)

    retriever.bm25 = FakeBM25()
    retriever.dense = FakeDense()
    retriever.fusion = ReciprocalRankFusion()

    result = retriever.retrieve("semiconductor")[0]

    assert result.chunk.metadata["bm25_rank"] == 1
    assert result.chunk.metadata["dense_rank"] == 1

    assert "bm25" in result.chunk.metadata["retrieval_sources"]
    assert "dense" in result.chunk.metadata["retrieval_sources"]