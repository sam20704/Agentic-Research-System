
from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    ):
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
#
        texts = [chunk.text for chunk in chunks]

# Real encoder returns (embeddings, attention_mask, token_type_ids).
# Fake encoder used in unit tests returns (embeddings, attention_mask).
        encoded = self.encoder.encode_documents(texts)

        if len(encoded) == 2:
            embeddings, masks = encoded
        elif len(encoded) == 3:
            embeddings, masks, _ = encoded
        else:
            raise ValueError(
        "encode_documents() must return (embeddings, mask) "
        "or (embeddings, mask, token_type_ids)."
        )
#
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

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search indexed chunks using ColBERT MaxSim."""

        if not query.strip():
            raise ValueError("query must not be empty")

        if self.count() == 0:
            return []

        top_k = top_k or self.config.default_top_k

        query_embeddings, query_mask = self.encoder.encode_query(query)

        query_embeddings = query_embeddings.cpu()
        query_mask = query_mask.cpu()

        scored: list[tuple[float, ColBERTIndexedChunk]] = []

        for indexed in self._index:
            score = self.encoder.maxsim_score(
                query_embeddings=query_embeddings,
                query_mask=query_mask,
                document_embeddings=indexed.token_embeddings,
                document_mask=indexed.attention_mask,
            )

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