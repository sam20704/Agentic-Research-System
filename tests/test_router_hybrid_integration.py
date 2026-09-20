from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import DocumentChunk, RetrievalResult
from src.retrieval.router import QueryComplexityRouter


def chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text="Semiconductor policy text",
        page_numbers=(1,),
        source="policy.pdf",
    )


class FakeBM25:
    def search(self, query, top_k):
        return [
            RetrievalResult(
                chunk=chunk("chunk1"),
                score=2.0,
                rank=1,
                retrieval_method="bm25",
            )
        ]


class FakeDense:
    def retrieve(self, query, top_k):
        return [
            RetrievalResult(
                chunk=chunk("chunk1"),
                score=0.95,
                rank=1,
                retrieval_method="dense",
            )
        ]


class FakeColBERT:
    def __init__(self):
        self.called = False

    def retrieve(self, query, top_k):
        self.called = True

        return [
            RetrievalResult(
                chunk=chunk("chunk2"),
                score=0.99,
                rank=1,
                retrieval_method="colbert",
            )
        ]


def test_router_enables_colbert():
    colbert = FakeColBERT()

    retriever = HybridRetriever(
        bm25_retriever=FakeBM25(),
        vectorstore=None,
        embedder=None,
        dense_retriever=FakeDense(),
        fusion=ReciprocalRankFusion(),
        colbert_retriever=colbert,
    )

    router = QueryComplexityRouter()

    query = (
        'Explain "trusted foundry" eligibility criteria '
        "for semiconductor manufacturing incentives."
    )

    decision = router.route(query)

    retriever.retrieve(
        query,
        use_colbert=decision.is_complex,
    )

    assert decision.is_complex is True
    assert colbert.called is True


def test_router_skips_colbert():
    colbert = FakeColBERT()

    retriever = HybridRetriever(
        bm25_retriever=FakeBM25(),
        vectorstore=None,
        embedder=None,
        dense_retriever=FakeDense(),
        fusion=ReciprocalRankFusion(),
        colbert_retriever=colbert,
    )

    router = QueryComplexityRouter()

    query = "apple"

    decision = router.route(query)

    retriever.retrieve(
        query,
        use_colbert=decision.is_complex,
    )

    assert decision.is_complex is False
    assert colbert.called is False