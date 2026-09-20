"""ColBERT late-interaction retrieval package."""

from .config import ColBERTConfig
from .encoder import ColBERTEncoder
from .index import ColBERTIndex
from .retriever import ColBERTRetriever

# Router now lives in src/retrieval/router
from src.retrieval.router import (
    QueryComplexityRouter,
    QueryRoutingDecision,
)

__all__ = [
    "ColBERTConfig",
    "ColBERTEncoder",
    "ColBERTIndex",
    "ColBERTRetriever",
    "QueryComplexityRouter",
    "QueryRoutingDecision",
]