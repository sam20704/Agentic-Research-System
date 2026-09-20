"""Hybrid sparse + dense retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

from src.retrieval.colbert.retriever import ColBERTRetriever
from src.retrieval.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.models import RetrievalResult
from src.retrieval.sparse.bm25 import BM25Retriever
from src.retrieval.vectorstore import QdrantVectorStore


class DenseRetriever:
    """Semantic retriever backed by Qdrant."""

    def __init__(
        self,
        vectorstore: QdrantVectorStore,
        embedder: SentenceTransformerEmbedding,
    ) -> None:
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """Retrieve semantic matches from Qdrant."""

        query_vector = self.embedder.embed_query(query)

        return self.vectorstore.search(
            query_vector=query_vector,
            top_k=top_k,
        )


class HybridRetriever:
    """Hybrid BM25 + dense retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vectorstore: QdrantVectorStore | None,
        embedder: SentenceTransformerEmbedding | None,
        fusion: ReciprocalRankFusion | None = None,
        dense_retriever: DenseRetriever | None = None,
        colbert_retriever: ColBERTRetriever | None = None,
        fusion_k: int = 60,
    ) -> None:
        self.bm25 = bm25_retriever
        self.colbert = colbert_retriever

        # Allow dependency injection for unit tests.
        if dense_retriever is not None:
            self.dense = dense_retriever
        else:
            if vectorstore is None or embedder is None:
                raise ValueError(
                    "vectorstore and embedder are required when "
                    "dense_retriever is not provided."
                )

            self.dense = DenseRetriever(
                vectorstore=vectorstore,
                embedder=embedder,
            )

        # Allow injecting a fake/custom fusion implementation in tests.
        self.fusion = (
            fusion
            if fusion is not None
            else ReciprocalRankFusion(k=fusion_k)
        )

    def retrieve(
        self,
        query: str,
        bm25_top_k: int = 20,
        dense_top_k: int = 20,
        final_top_k: int = 10,
        use_colbert: bool = False,
    ) -> list[RetrievalResult]:
        """
        Run BM25 retrieval, dense retrieval, then fuse the rankings.

        If `use_colbert=True` and a ColBERT retriever is configured,
        augment the fused candidate pool with ColBERT candidates before
        returning the final retrieval set.
        """

        bm25_results = self.bm25.search(
            query=query,
            top_k=bm25_top_k,
        )

        dense_results = self.dense.retrieve(
            query=query,
            top_k=dense_top_k,
        )

        fused_results = self.fusion.fuse(
            bm25_results,
            dense_results,
            top_k=max(final_top_k, dense_top_k),
        )

        # --------------------------------------------------------------
        # Optional Phase 3.1 ColBERT augmentation.
        # --------------------------------------------------------------
        if use_colbert and self.colbert is not None:
            colbert_results = self.colbert.retrieve(
                query=query,
                top_k=final_top_k,
            )

            fused_chunk_ids = {
                result.chunk.chunk_id
                for result in fused_results
            }

            for result in colbert_results:
                if result.chunk.chunk_id not in fused_chunk_ids:
                    fused_results.append(result)

        return fused_results[:final_top_k]