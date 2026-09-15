"""Embedding providers for the retrieval system."""

from src.retrieval.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)

__all__ = [
    "SentenceTransformerEmbedding",
]