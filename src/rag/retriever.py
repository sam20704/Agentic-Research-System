"""Hybrid RAG retriever (Phase 2.4).

Uses:
- BM25 sparse retrieval
- Qdrant dense retrieval (BGE-M3 embeddings)
- Reciprocal Rank Fusion (RRF)
- Existing query expansion + decomposition logic
"""

from __future__ import annotations

from src.retrieval.embeddings.sentence_transformer import BGEM3Embedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalResult
from src.retrieval.sparse.bm25 import BM25Retriever
from src.retrieval.vectorstore import QdrantVectorStore

# ---------------------------------------------------------------------
# Default Retrieval Configuration
# ---------------------------------------------------------------------

DEFAULT_TOP_K = 10
DEFAULT_BM25_TOP_K = 20
DEFAULT_DENSE_TOP_K = 20

# ---------------------------------------------------------------------
# Query Expansion
# ---------------------------------------------------------------------


def expand_query(query: str) -> str:
    """Light query expansion for policy/economy terminology."""
    expansion_terms = (
        " semiconductor policy incentives fiscal support subsidy "
        "government support manufacturing EV supply chain"
    )
    return query.strip() + expansion_terms


# ---------------------------------------------------------------------
# Query Decomposition
# ---------------------------------------------------------------------


def decompose_query(query: str) -> list[str]:
    """
    Conservative query decomposition.
    Only decomposes clearly multi-part questions.
    """

    q = query.strip()
    q_lower = q.lower()

    subqueries = [q]

    if (
        "compare" in q_lower
        and "india" in q_lower
        and "taiwan" in q_lower
        and "semiconductor" in q_lower
    ):
        subqueries.extend(
            [
                "India semiconductor policy",
                "Taiwan semiconductor policy",
            ]
        )

    elif (
        "compare" in q_lower
        and "india" in q_lower
        and "global" in q_lower
        and "ev" in q_lower
    ):
        subqueries.extend(
            [
                "India EV adoption trends",
                "Global EV adoption trends",
            ]
        )

    elif (
        (
            "analyze" in q_lower
            or "impact" in q_lower
            or "together" in q_lower
        )
        and "semiconductor" in q_lower
        and "ev" in q_lower
        and "india" in q_lower
    ):
        subqueries.extend(
            [
                "India semiconductor policy industrial growth",
                "India EV policy industrial growth",
            ]
        )

    elif any(word in q_lower for word in ["contradiction", "gap"]):
        subqueries.extend(
            [
                "policy goals and adoption trends India EV",
                "policy goals and adoption trends India semiconductor",
            ]
        )

    elif "risk" in q_lower and "supply chain" in q_lower:
        subqueries.extend(
            [
                "EV supply chain risks",
                "Semiconductor supply chain risks",
            ]
        )

    elif (
        "challenge" in q_lower
        and "india" in q_lower
        and "semiconductor" in q_lower
    ):
        subqueries.extend(
            [
                "India semiconductor manufacturing challenges",
                "barriers to semiconductor manufacturing in India",
            ]
        )

    # Remove duplicates while preserving order.
    seen = set()
    final_queries = []

    for sq in subqueries:
        key = sq.lower().strip()
        if key not in seen:
            seen.add(key)
            final_queries.append(sq.strip())

    return final_queries


# ---------------------------------------------------------------------
# Hybrid Retriever Wrapper
# ---------------------------------------------------------------------


class RAGRetriever:
    """
    High-level retrieval interface for the RAG pipeline.

    Wraps the HybridRetriever while preserving the existing public API.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vectorstore: QdrantVectorStore,
        embedder: BGEM3Embedder,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        final_top_k: int = DEFAULT_TOP_K,
    ) -> None:

        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k

        self.hybrid = HybridRetriever(
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
        Hybrid retrieval with query decomposition and RRF fusion.
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        top_k = top_k or self.final_top_k

        subqueries = decompose_query(query)

        if verbose:
            print("=" * 60)
            print("HYBRID RETRIEVAL")
            print("=" * 60)
            print(f"Original Query : {query}")
            print(f"Subqueries ({len(subqueries)}):")
            for i, sq in enumerate(subqueries, start=1):
                print(f"  {i}. {sq}")

        # -------------------------------------------------------------
        # FAST PATH (Phase 2.4 optimization)
        # Most queries generate only one subquery.
        # Avoid repeated BM25 + Dense retrieval + RRF.
        # -------------------------------------------------------------
        if len(subqueries) == 1:
            results = self.hybrid.retrieve(
                query=expand_query(subqueries[0]),
                bm25_top_k=self.bm25_top_k,
                dense_top_k=self.dense_top_k,
                final_top_k=top_k,
            )

            if return_metadata:
                return results

            return [result.chunk.text for result in results]

        # -------------------------------------------------------------
        # Multi-query retrieval
        # -------------------------------------------------------------
        all_results = []

        for sq in subqueries:
            expanded_query = expand_query(sq)

            results = self.hybrid.retrieve(
                query=expanded_query,
                bm25_top_k=self.bm25_top_k,
                dense_top_k=self.dense_top_k,
                final_top_k=top_k,
            )

            all_results.extend(results)

        # -------------------------------------------------------------
        # Deduplicate by chunk_id, keeping the highest scoring result.
        # -------------------------------------------------------------
        merged: dict[str, RetrievalResult] = {}

        for result in all_results:
            chunk_id = result.chunk.chunk_id

            if (
                chunk_id not in merged
                or result.score > merged[chunk_id].score
            ):
                merged[chunk_id] = result

        ordered_results = sorted(
            merged.values(),
            key=lambda r: r.score,
            reverse=True,
        )[:top_k]

        # -------------------------------------------------------------
        # RetrievalResult is a frozen dataclass.
        # Create new RetrievalResult objects with updated ranks.
        # -------------------------------------------------------------
        final_results = [
            RetrievalResult(
                chunk=result.chunk,
                score=result.score,
                rank=rank,
                retrieval_method=result.retrieval_method,
            )
            for rank, result in enumerate(ordered_results, start=1)
        ]

        if verbose:
            print("\nFinal Results")
            print("-" * 60)

            for result in final_results:
                print(
                    f"[{result.rank}] "
                    f"{result.chunk.chunk_id} "
                    f"Score={result.score:.4f}"
                )
                print(f"Source : {result.chunk.source}")
                print(f"Pages  : {result.chunk.page_numbers}")
                print(
                    f"Methods: "
                    f"{result.chunk.metadata.get('retrieval_sources', [])}"
                )
                print(result.chunk.text[:180].replace("\n", " "))
                print()

        if return_metadata:
            return final_results

        # Preserve the previous public API.
        return [result.chunk.text for result in final_results]


# ---------------------------------------------------------------------
# Convenience Global Retriever
# ---------------------------------------------------------------------

_retriever: RAGRetriever | None = None


def configure_retriever(
    bm25_retriever: BM25Retriever,
    vectorstore: QdrantVectorStore,
    embedder: BGEM3Embedder,
) -> None:
    """
    Configure a global retriever instance used by retrieve().
    Call once during application startup after indexing.
    """

    global _retriever

    _retriever = RAGRetriever(
        bm25_retriever=bm25_retriever,
        vectorstore=vectorstore,
        embedder=embedder,
    )


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = False,
    return_metadata: bool = False,
):
    """
    Backward-compatible retrieval function.

    Existing code can continue calling:
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