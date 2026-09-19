from __future__ import annotations

from collections import defaultdict

from src.retrieval.models import RetrievalResult


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF).

    Score(chunk) = Σ 1 / (k + rank)
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        *rankings: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:

        fused_scores = defaultdict(float)
        fused_results: dict[str, RetrievalResult] = {}

        provenance = defaultdict(
            lambda: {
                "bm25_rank": None,
                "dense_rank": None,
                "retrieval_sources": [],
            }
        )

        # Collect scores from each retrieval ranking.
        for ranking in rankings:

            if not ranking:
                continue

            method = ranking[0].retrieval_method

            for rank, result in enumerate(ranking, start=1):

                chunk_id = result.chunk.chunk_id

                fused_scores[chunk_id] += 1 / (self.k + rank)

                # Keep the first RetrievalResult for this chunk.
                fused_results.setdefault(chunk_id, result)

                if method == "bm25":
                    provenance[chunk_id]["bm25_rank"] = rank

                elif method == "dense":
                    provenance[chunk_id]["dense_rank"] = rank

                provenance[chunk_id]["retrieval_sources"].append(method)

        # Order by fused score.
        ordered = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        fused: list[RetrievalResult] = []

        for final_rank, (chunk_id, score) in enumerate(
            ordered[:top_k],
            start=1,
        ):

            original_result = fused_results[chunk_id]

            # Create new metadata instead of mutating frozen dataclass.
            metadata = {
                **original_result.chunk.metadata,
                **provenance[chunk_id],
                "rrf_score": score,
            }

            # Create a new DocumentChunk with updated metadata.
            updated_chunk = original_result.chunk.__class__(
                chunk_id=original_result.chunk.chunk_id,
                document_id=original_result.chunk.document_id,
                text=original_result.chunk.text,
                page_numbers=original_result.chunk.page_numbers,
                source=original_result.chunk.source,
                section=original_result.chunk.section,
                bounding_boxes=original_result.chunk.bounding_boxes,
                element_ids=original_result.chunk.element_ids,
                metadata=metadata,
            )

            # Create a new RetrievalResult (don't mutate frozen instance).
            fused.append(
                RetrievalResult(
                    chunk=updated_chunk,
                    score=score,
                    rank=final_rank,
                    retrieval_method="hybrid",
                )
            )

        return fused