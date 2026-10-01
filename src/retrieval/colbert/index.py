"""Deterministic in-memory ColBERT index for Phase 3.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.retrieval.colbert.config import ColBERTConfig
from src.retrieval.colbert.encoder import ColBERTEncoder
from src.retrieval.models import DocumentChunk, RetrievalResult


@dataclass(frozen=True)
class ColBERTIndexedChunk:
    """Token-level ColBERT representation of one document chunk."""

    chunk: DocumentChunk
    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor


class ColBERTIndex:
    """
    Deterministic in-memory ColBERT index.

    Phase 3.1 intentionally keeps the index implementation isolated
    from Qdrant, BM25, RRF, routing, and reranking.
    """

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
        """
        Build a deterministic token-level index.

        Chunks are sorted by chunk_id before encoding so index
        construction does not depend on caller ordering.
        """

        if not chunks:
            self._index = []
            return

        ordered_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.chunk_id,
        )

        chunk_ids = [
            chunk.chunk_id
            for chunk in ordered_chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "ColBERT index requires unique chunk_id values."
            )

        texts = [
            chunk.text
            for chunk in ordered_chunks
        ]

        encoded = self.encoder.encode_documents(texts)

        if isinstance(encoded, tuple):
            embeddings, masks = encoded[:2]
        else:
            embeddings = encoded
            masks = [
                torch.ones(
                    embedding.shape[0],
                    dtype=torch.bool,
                )
                for embedding in embeddings
            ]

        if len(embeddings) != len(ordered_chunks):
            raise RuntimeError(
                "Number of document embeddings does not match "
                "number of chunks."
            )

        if len(masks) != len(ordered_chunks):
            raise RuntimeError(
                "Number of document masks does not match "
                "number of chunks."
            )

        self._index = [
            ColBERTIndexedChunk(
                chunk=chunk,
                token_embeddings=embedding.detach().cpu(),
                attention_mask=mask.detach().cpu(),
            )
            for chunk, embedding, mask in zip(
                ordered_chunks,
                embeddings,
                masks,
                strict=True,
            )
        ]

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return len(self._index)

    def size(self) -> int:
        """Return the number of indexed chunks."""

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
            raise ValueError("query must not be empty.")

        if self.count() == 0:
            return []

        top_k = (
            self.config.default_top_k
            if top_k is None
            else top_k
        )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_encoded = self.encoder.encode_query(query)

        if isinstance(query_encoded, tuple):
            query_embeddings, query_mask = query_encoded[:2]
        else:
            query_embeddings = query_encoded
            query_mask = torch.ones(
                query_embeddings.shape[0],
                dtype=torch.bool,
            )

        query_embeddings = query_embeddings.detach().cpu()
        query_mask = query_mask.detach().cpu()

        scored: list[tuple[float, ColBERTIndexedChunk]] = []

        for indexed in self._index:
            score = self.encoder.maxsim_score(
                query_embeddings=query_embeddings,
                query_mask=query_mask,
                document_embeddings=indexed.token_embeddings,
                document_mask=indexed.attention_mask,
            )

            scored.append(
                (
                    float(score),
                    indexed,
                )
            )

        # Deterministic ordering:
        #   1. higher ColBERT score
        #   2. chunk_id lexical order for ties
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].chunk.chunk_id,
            )
        )

        results: list[RetrievalResult] = []

        for rank, (score, indexed) in enumerate(
            scored[:top_k],
            start=1,
        ):
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
                section=indexed.chunk.section,
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