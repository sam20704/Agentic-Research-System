
from __future__ import annotations

from src.retrieval.embeddings.sentence_transformer import BGEM3Embedder
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.models import RetrievalResult
from src.retrieval.sparse.bm25 import BM25Retriever
from src.retrieval.vectorstore import QdrantVectorStore


class DenseRetriever:
    """Semantic retriever backed by Qdrant."""

    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        embedder: BGEM3Embedder,
    ):
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:

        query_vector = self.embedder.embed_query(query)

        return self.vectorstore.search(
            query_vector=query_vector,
            top_k=top_k,
        )


class HybridRetriever:
    """Hybrid sparse + dense retriever using RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vectorstore: QdrantVectorStore,
        embedder: BGEM3Embedder,
        fusion_k: int = 60,
    ):
        self.bm25 = bm25_retriever
        self.dense = DenseRetriever(
            vectorstore=vectorstore,
            embedder=embedder,
        )

        self.fusion = ReciprocalRankFusion(k=fusion_k)

    def retrieve(
        self,
        query: str,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        final_top_k: int = 10,
    ) -> list[RetrievalResult]:

        bm25_results = self.bm25.search(
            query=query,
            top_k=bm25_top_k,
        )

        dense_results = self.dense.retrieve(
            query=query,
            top_k=dense_top_k,
        )

        return self.fusion.fuse(
            bm25_results,
            dense_results,
            top_k=final_top_k,
        )