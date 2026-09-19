from src.rag.retriever import configure_retriever, retrieve
from src.retrieval.models import DocumentChunk, RetrievalResult


class FakeEmbedder:
    def embed_query(self, query):
        return [0.0] * 1024


class FakeVectorStore:
    def search(self, query_vector, top_k):
        chunk = DocumentChunk(
            chunk_id="chunk1",
            document_id="doc1",
            text="India semiconductor policy provides fiscal incentives.",
            page_numbers=(1,),
            source="policy.pdf",
            metadata={},
        )
        return [
            RetrievalResult(
                chunk=chunk,
                score=0.95,
                rank=1,
                retrieval_method="dense",
            )
        ]


class FakeBM25:
    def search(self, query, top_k):
        chunk = DocumentChunk(
            chunk_id="chunk1",
            document_id="doc1",
            text="India semiconductor policy provides fiscal incentives.",
            page_numbers=(1,),
            source="policy.pdf",
            metadata={},
        )
        return [
            RetrievalResult(
                chunk=chunk,
                score=8.2,
                rank=1,
                retrieval_method="bm25",
            )
        ]


def test_retriever_returns_results():
    configure_retriever(
        bm25_retriever=FakeBM25(),
        vectorstore=FakeVectorStore(),
        embedder=FakeEmbedder(),
    )

    results = retrieve("What is semiconductor policy in India?")

    assert isinstance(results, list)
    assert len(results) == 1