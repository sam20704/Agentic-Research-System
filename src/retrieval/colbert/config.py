
"""Configuration for ColBERT late-interaction retrieval (Phase 3.1)."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_QUERY_LENGTH = 64
DEFAULT_MAX_DOCUMENT_LENGTH = 180
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class ColBERTConfig:
    """
    Configuration for the ColBERT retrieval pipeline.

    The configuration is immutable so retrieval behavior remains
    deterministic once a retriever has been constructed.
    """

    model_name: str = DEFAULT_COLBERT_MODEL
    device: str = "auto"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_query_length: int = DEFAULT_MAX_QUERY_LENGTH
    max_document_length: int = DEFAULT_MAX_DOCUMENT_LENGTH
    default_top_k: int = DEFAULT_TOP_K

    def __post_init__(self) -> None:
        """Validate configuration values."""

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.max_query_length <= 0:
            raise ValueError("max_query_length must be greater than zero.")

        if self.max_document_length <= 0:
            raise ValueError("max_document_length must be greater than zero.")

        if self.default_top_k <= 0:
            raise ValueError("default_top_k must be greater than zero.")

        allowed_devices = {"auto", "cpu", "cuda", "mps"}

        if self.device not in allowed_devices:
            raise ValueError(
                f"Unsupported device '{self.device}'. "
                f"Choose from {sorted(allowed_devices)}."
            )