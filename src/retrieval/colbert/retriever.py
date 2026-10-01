
"""Independent ColBERT late-interaction retriever for Phase 3.1."""

from __future__ import annotations

from src.retrieval.colbert.config import ColBERTConfig
from src.retrieval.colbert.encoder import ColBERTEncoder
from src.retrieval.colbert.index import ColBERTIndex
from src.retrieval.models import DocumentChunk, RetrievalResult


class ColBERTRetriever:
    """
    Standalone ColBERT late-interaction retriever.

    Public lifecycle:

        retriever = ColBERTRetriever(...)
        retriever.build_index(chunks)
        results = retriever.retrieve(query, top_k=10)

    This class intentionally has no dependency on:

    - BM25
    - BGE-M3
    - Qdrant
    - RRF
    - reranking
    - query routing
    - agent orchestration
    """

    def __init__(
        self,
        config: ColBERTConfig | None = None,
        encoder: ColBERTEncoder | None = None,
        index: ColBERTIndex | None = None,
    ) -> None:
        self.config = config or ColBERTConfig()

        self.encoder = (
            encoder
            if encoder is not None
            else ColBERTEncoder(self.config)
        )

        self.index = (
            index
            if index is not None
            else ColBERTIndex(
                encoder=self.encoder,
                config=self.config,
            )
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """Encode and index document chunks."""

        self.index.build(chunks)

    def index_size(self) -> int:
        """Return the number of indexed chunks."""

        return self.index.size()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k chunks using ColBERT late interaction."""

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        top_k = (
            self.config.default_top_k
            if top_k is None
            else top_k
        )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return self.index.search(
            query=query,
            top_k=top_k,
        )