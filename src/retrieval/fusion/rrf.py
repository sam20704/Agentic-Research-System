from __future__ import annotations

from collections import defaultdict

from src.retrieval.models import RetrievalResult


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF).

    RRF combines ranked retrieval results without comparing raw scores.

    Score(chunk) = Σ 1 / (k + rank)

    Contract:
        Each positional ranking passed to `fuse()` must contain results
        from exactly one retrieval method (for example BM25 or Dense).
    """

    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.k = k

    def fuse(
        self,
        *rankings: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Fuse multiple ranked retrieval lists into one ranked list.

        Parameters
        ----------
        rankings:
            Ranked RetrievalResult lists from different retrieval methods.

        top_k:
            Maximum number of fused results to return.
        """

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        fused_scores = defaultdict(float)
        fused_results: dict[str, RetrievalResult] = {}

        provenance = defaultdict(
            lambda: {
                "bm25_rank": None,
                "dense_rank": None,
                "retrieval_sources": [],
            }
        )

        for ranking in rankings:
            if not ranking:
                continue

            # ---------------------------------------------------------
            # Every ranking must come from exactly one retrieval method.
            # ---------------------------------------------------------
            methods = {r.retrieval_method for r in ranking}

            if len(methods) != 1:
                raise ValueError(
                    "Each ranking must contain results from exactly one retrieval method."
                )

            method = methods.pop()

            for rank, result in enumerate(ranking, start=1):
                chunk_id = result.chunk.chunk_id

                fused_scores[chunk_id] += 1 / (self.k + rank)

                fused_results.setdefault(chunk_id, result)

                if method == "bm25":
                    provenance[chunk_id]["bm25_rank"] = rank

                elif method == "dense":
                    provenance[chunk_id]["dense_rank"] = rank

                provenance[chunk_id]["retrieval_sources"].append(method)

        # -------------------------------------------------------------
        # Deterministic ordering:
        #   1. Higher RRF score.
        #   2. Lexicographically smaller chunk_id.
        # -------------------------------------------------------------
        ordered = sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        fused: list[RetrievalResult] = []

        for final_rank, (chunk_id, score) in enumerate(
            ordered[:top_k],
            start=1,
        ):
            result = fused_results[chunk_id]

            metadata = {
                **result.chunk.metadata,
                **provenance[chunk_id],
                "rrf_score": score,
            }

            updated_chunk = result.chunk.__class__(
                chunk_id=result.chunk.chunk_id,
                document_id=result.chunk.document_id,
                text=result.chunk.text,
                page_numbers=result.chunk.page_numbers,
                source=result.chunk.source,
                section=result.chunk.section,
                bounding_boxes=result.chunk.bounding_boxes,
                element_ids=result.chunk.element_ids,
                metadata=metadata,
            )

            fused.append(
                RetrievalResult(
                    chunk=updated_chunk,
                    score=score,
                    rank=final_rank,
                    retrieval_method="hybrid",
                )
            )

        return fused