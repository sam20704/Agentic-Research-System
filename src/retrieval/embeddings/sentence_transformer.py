"""Sentence Transformer based embedding provider."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.models import DocumentChunk


class SentenceTransformerEmbedding:
    """Generate dense embeddings using a Sentence Transformer model.

    The embedding provider is intentionally independent of the vector
    database. It converts text or DocumentChunk objects into dense vectors
    suitable for semantic retrieval.
    """

    DEFAULT_MODEL = "BAAI/bge-m3"
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_MAX_SEQ_LENGTH = 1024

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty")

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if max_seq_length < 1:
            raise ValueError("max_seq_length must be >= 1")

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self._model = SentenceTransformer(
            model_name,
            device=device,
        )

        self._model.max_seq_length = max_seq_length

        dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension"
            )

        self._dimension = int(dimension)

        if self._dimension <= 0:
            raise RuntimeError(
                "Embedding dimension must be positive"
            )

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    # ------------------------------------------------------------------
    # Single text embedding
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""

        if not text or not text.strip():
            raise ValueError("text must not be empty")

        embedding = self._model.encode(
            text,
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        self._validate_vector(vector)

        return vector.tolist()

    # ------------------------------------------------------------------
    # Phase 2.4 Compatibility Method
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a search query.

        HybridRetriever expects an `embed_query()` method, so this simply
        delegates to `embed_text()`.
        """
        return self.embed_text(query)

    # ------------------------------------------------------------------
    # Batch text embeddings
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

        if not texts:
            return []

        text_list = list(texts)

        if any(not text or not text.strip() for text in text_list):
            raise ValueError(
                "texts must not contain empty values"
            )

        effective_batch_size = (
            self.batch_size
            if batch_size is None
            else batch_size
        )

        if effective_batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1"
            )

        embeddings = self._model.encode(
            text_list,
            batch_size=effective_batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise RuntimeError(
                "Expected a 2D embedding matrix"
            )

        if matrix.shape[0] != len(text_list):
            raise RuntimeError(
                "Embedding count does not match input count"
            )

        if matrix.shape[1] != self.dimension:
            raise RuntimeError(
                "Embedding dimension does not match model dimension"
            )

        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(
                "Embeddings contain non-finite values"
            )

        return matrix.tolist()

    # ------------------------------------------------------------------
    # DocumentChunk embeddings
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for DocumentChunk objects."""

        if not chunks:
            return []

        return self.embed_texts(
            [chunk.text for chunk in chunks],
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_vector(self, vector: np.ndarray) -> None:
        """Validate a single embedding vector."""

        if vector.ndim != 1:
            raise RuntimeError(
                "Expected a one-dimensional embedding vector"
            )

        if vector.shape[0] != self.dimension:
            raise RuntimeError(
                "Embedding dimension does not match model dimension"
            )

        if not np.all(np.isfinite(vector)):
            raise RuntimeError(
                "Embedding contains non-finite values"
            )

        if self.normalize_embeddings:
            norm = float(np.linalg.norm(vector))

            if not math.isclose(
                norm,
                1.0,
                rel_tol=1e-4,
                abs_tol=1e-4,
            ):
                raise RuntimeError(
                    "Normalized embedding does not have unit norm"
                )

