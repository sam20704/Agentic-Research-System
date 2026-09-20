"""ColBERT late-interaction retriever."""

from __future__ import annotations

from src.retrieval.colbert.config import ColBERTConfig
from src.retrieval.colbert.encoder import ColBERTEncoder
from src.retrieval.colbert.index import ColBERTIndex
from src.retrieval.models import DocumentChunk, RetrievalResult


class ColBERTRetriever:
    """
    ColBERT retriever using token-level late interaction.

    This implementation is additive to the existing Phase 2 retrieval
    pipeline and returns RetrievalResult objects compatible with BM25,
    dense retrieval, RRF, and reranking.
    """

    def __init__(
        self,
        config: ColBERTConfig | None = None,
        encoder: ColBERTEncoder | None = None,
        index: ColBERTIndex | None = None,
    ):
        self.config = config or ColBERTConfig()
        self.encoder = encoder or ColBERTEncoder(self.config)
        self.index = index or ColBERTIndex(self.encoder, self.config)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Build the ColBERT index from document chunks."""
        self.index.build(chunks)

    def index_size(self) -> int:
        """Return number of indexed chunks."""
        return self.index.count()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve ranked evidence using ColBERT MaxSim scoring.
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        top_k = top_k or self.config.default_top_k

        return self.index.search(
            query=query,
            top_k=top_k,
        )