"""Sentence Transformer based embedding provider."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.models import DocumentChunk


class SentenceTransformerEmbedding:
    """Generate embeddings using a Sentence Transformer model.

    The class is intentionally independent of the vector database.
    It converts text or DocumentChunk objects into dense vectors.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self._model = SentenceTransformer(
            model_name,
            device=device,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension"
            )

        return int(dimension)

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        ).tolist()

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError(
                "texts must not contain empty values"
            )

        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        ).tolist()

    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate embeddings for DocumentChunk objects."""

        return self.embed_texts(
            [chunk.text for chunk in chunks],
            batch_size=batch_size,
        )