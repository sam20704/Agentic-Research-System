
"""ColBERT late-interaction retriever."""

from __future__ import annotations

from src.retrieval.colbert.config import ColBERTConfig
from src.retrieval.colbert.encoder import ColBERTEncoder
from src.retrieval.colbert.index import ColBERTIndex
from src.retrieval.models import DocumentChunk, RetrievalResult


class ColBERTRetriever:
    """
    ColBERT late-interaction retriever.

    Supports dependency injection for unit tests while using the real
    ColBERT encoder/index in integration tests.
    """

    def __init__(
        self,
        config: ColBERTConfig | None = None,
        encoder: ColBERTEncoder | None = None,
        index: ColBERTIndex | None = None,
    ) -> None:
        self.config = config or ColBERTConfig()

        # Allow fake encoder/index injection for unit tests.
        self.encoder = encoder or ColBERTEncoder(self.config)
        self.index = index or ColBERTIndex(self.encoder)

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
        """Return number of indexed chunks."""
        return self.index.size()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k document chunks using late interaction.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        top_k = top_k or self.config.default_top_k

        # Let the index encode the query (keeps one canonical interface).
        return self.index.search(
            query=query,
            top_k=top_k,
        )