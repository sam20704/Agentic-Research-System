"""Integration tests for real ColBERT length configuration."""

from __future__ import annotations

import os

import pytest

from src.retrieval.colbert.config import ColBERTConfig
from src.retrieval.colbert.encoder import ColBERTEncoder


RUN_PHASE3_COLBERT_INTEGRATION_TESTS = (
    os.getenv("RUN_PHASE3_COLBERT_INTEGRATION_TESTS") == "1"
)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_PHASE3_COLBERT_INTEGRATION_TESTS,
    reason=(
        "Set RUN_PHASE3_COLBERT_INTEGRATION_TESTS=1 "
        "to run real ColBERT integration tests."
    ),
)
def test_configured_query_and_document_lengths_reach_transformer():
    config = ColBERTConfig(
        device="cpu",
        batch_size=2,
        max_query_length=32,
        max_document_length=96,
        default_top_k=10,
    )

    encoder = ColBERTEncoder(config)

    transformer_modules = [
        module
        for module in encoder.model
        if module.__class__.__name__ == "Transformer"
    ]

    assert len(transformer_modules) == 1

    transformer = transformer_modules[0]

    assert transformer.query_length == 32
    assert transformer.document_length == 96
