"""Backward-compatible RAG retriever adapter (Phase 3.1).

The canonical retrieval implementation lives in src/retrieval/hybrid.py.

This module preserves the existing retrieve() API while delegating all
retrieval logic to HybridRetriever and introducing an optional
Query Complexity Router for ColBERT augmentation.
"""

from __future__ import annotations

from src.retrieval.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.router import QueryComplexityRouter
from src.retrieval.sparse.bm25 import BM25Retriever
from src.retrieval.vectorstore import QdrantVectorStore

# ---------------------------------------------------------------------
# Default Retrieval Configuration
# ---------------------------------------------------------------------

DEFAULT_TOP_K = 10
DEFAULT_BM25_TOP_K = 20
DEFAULT_DENSE_TOP_K = 20


# ---------------------------------------------------------------------
# Backward-Compatible RAG Retriever
# ---------------------------------------------------------------------


class RAGRetriever:
    """
    Thin compatibility adapter around HybridRetriever.

    Phase 3.1 canonical retrieval flow:

        Query
          ↓
    Query Complexity Router
          ↓
    BM25 + Dense (+ Optional ColBERT)
          ↓
            RRF
          ↓
    RetrievalResult[]
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vectorstore: QdrantVectorStore,
        embedder: SentenceTransformerEmbedding,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
    ) -> None:
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k

        self.router = QueryComplexityRouter()

        self.retriever = HybridRetriever(
            bm25_retriever=bm25_retriever,
            vectorstore=vectorstore,
            embedder=embedder,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        verbose: bool = False,
        return_metadata: bool = False,
    ):
        """
        Delegate retrieval to the canonical HybridRetriever.

        Phase 3.1 adds automatic routing for complex queries so that
        ColBERT can augment the hybrid retrieval pipeline when available.
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        # Preserve legacy behavior where callers may pass top_k=None.
        top_k = top_k or DEFAULT_TOP_K

        # --------------------------------------------------------------
        # Phase 3.1 Query Router
        # --------------------------------------------------------------
        decision = self.router.route(query)

        results = self.retriever.retrieve(
            query=query,
            bm25_top_k=self.bm25_top_k,
            dense_top_k=self.dense_top_k,
            final_top_k=top_k,
            use_colbert=decision.is_complex,
        )

        if verbose:
            print("=" * 60)
            print("HYBRID RETRIEVAL")
            print("=" * 60)
            print(f"Query           : {query}")
            print(f"Router Decision : {decision.reason}")
            print(f"Use ColBERT     : {decision.is_complex}")
            print()

            for result in results:
                print(
                    f"[{result.rank}] "
                    f"{result.chunk.chunk_id} "
                    f"Score={result.score:.4f}"
                )
                print(f"Source : {result.chunk.source}")
                print(f"Pages  : {result.chunk.page_numbers}")
                print(
                    "Methods: "
                    f"{result.chunk.metadata.get('retrieval_sources', [])}"
                )
                print(result.chunk.text[:180].replace("\n", " "))
                print()

        if return_metadata:
            return results

        # Preserve the legacy API: return only chunk text.
        return [result.chunk.text for result in results]


# ---------------------------------------------------------------------
# Global Retriever Instance (Backward Compatibility)
# ---------------------------------------------------------------------

_retriever: RAGRetriever | None = None


def configure_retriever(
    bm25_retriever: BM25Retriever,
    vectorstore: QdrantVectorStore,
    embedder: SentenceTransformerEmbedding,
) -> None:
    """
    Configure the global retriever instance.

    Call this once during application startup after indexing the corpus.
    """

    global _retriever

    _retriever = RAGRetriever(
        bm25_retriever=bm25_retriever,
        vectorstore=vectorstore,
        embedder=embedder,
    )


def retrieve(
    query: str,
    top_k: int | None = None,
    verbose: bool = False,
    return_metadata: bool = False,
):
    """
    Backward-compatible retrieval entry point.

    Existing generator code can continue calling:

        retrieve("your query")
    """

    if _retriever is None:
        raise RuntimeError(
            "Retriever has not been configured. "
            "Call configure_retriever(...) during startup."
        )

    return _retriever.retrieve(
        query=query,
        top_k=top_k,
        verbose=verbose,
        return_metadata=return_metadata,
    )