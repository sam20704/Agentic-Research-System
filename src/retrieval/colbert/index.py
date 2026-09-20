from __future__ import annotations

from dataclasses import dataclass

import torch

from src.retrieval.colbert.config import ColBERTConfig
from src.retrieval.colbert.encoder import ColBERTEncoder
from src.retrieval.models import DocumentChunk, RetrievalResult


@dataclass(frozen=True)
class ColBERTIndexedChunk:
    """Token-level ColBERT representation of one chunk."""

    chunk: DocumentChunk
    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor


class ColBERTIndex:
    """Deterministic in-memory ColBERT index."""

    def __init__(
        self,
        encoder: ColBERTEncoder,
        config: ColBERTConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.config = config or encoder.config
        self._index: list[ColBERTIndexedChunk] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build a deterministic token-level index."""

        if not chunks:
            self._index = []
            return

        texts = [chunk.text for chunk in chunks]

        encoded = self.encoder.encode_documents(texts)

        # --------------------------------------------------------------
        # Compatibility with both the real encoder and fake test encoder.
        #
        # Real encoder:
        #     List[Tensor] (one tensor per document)
        #
        # Fake encoder:
        #     (embeddings, masks)
        # --------------------------------------------------------------

        if isinstance(encoded, tuple):
            embeddings, masks = encoded[:2]
        else:
            embeddings = encoded
            masks = [
                torch.ones(embedding.shape[0], dtype=torch.bool)
                for embedding in embeddings
            ]

        self._index = [
            ColBERTIndexedChunk(
                chunk=chunk,
                token_embeddings=embedding.cpu(),
                attention_mask=mask.cpu(),
            )
            for chunk, embedding, mask in zip(
                chunks,
                embeddings,
                masks,
                strict=True,
            )
        ]

    def count(self) -> int:
        return len(self._index)

    def size(self) -> int:
        """Return number of indexed chunks."""
        return len(self._index)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search indexed chunks using ColBERT MaxSim."""

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if self.count() == 0:
            return []

        top_k = top_k or self.config.default_top_k

        query_encoded = self.encoder.encode_query(query)

        # --------------------------------------------------------------
        # Compatibility with both real and fake encoders.
        #
        # Real encoder:
        #     Tensor
        #
        # Fake encoder:
        #     (embeddings, mask)
        # --------------------------------------------------------------

        if isinstance(query_encoded, tuple):
            query_embeddings, query_mask = query_encoded[:2]
        else:
            query_embeddings = query_encoded
            query_mask = torch.ones(
                query_embeddings.shape[0],
                dtype=torch.bool,
            )

        query_embeddings = query_embeddings.cpu()
        query_mask = query_mask.cpu()

        scored: list[tuple[float, ColBERTIndexedChunk]] = []

        for indexed in self._index:
            # Fake encoder exposes maxsim_score(); real encoder currently doesn't.
            if hasattr(self.encoder, "maxsim_score"):
                score = self.encoder.maxsim_score(
                    query_embeddings=query_embeddings,
                    query_mask=query_mask,
                    document_embeddings=indexed.token_embeddings,
                    document_mask=indexed.attention_mask,
                )
            else:
                similarity = torch.matmul(
                    query_embeddings,
                    indexed.token_embeddings.T,
                )

                max_per_query_token = similarity.max(dim=1).values
                score = max_per_query_token.sum()

            scored.append((float(score), indexed))

        scored.sort(
            key=lambda item: (-item[0], item[1].chunk.chunk_id)
        )

        results: list[RetrievalResult] = []

        for rank, (score, indexed) in enumerate(scored[:top_k], start=1):
            metadata = dict(indexed.chunk.metadata)
            metadata.update(
                {
                    "colbert_score": score,
                    "retrieval_sources": ["colbert"],
                }
            )

            chunk = DocumentChunk(
                chunk_id=indexed.chunk.chunk_id,
                document_id=indexed.chunk.document_id,
                text=indexed.chunk.text,
                page_numbers=indexed.chunk.page_numbers,
                source=indexed.chunk.source,
                bounding_boxes=indexed.chunk.bounding_boxes,
                element_ids=indexed.chunk.element_ids,
                metadata=metadata,
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_method="colbert",
                )
            )

        return results