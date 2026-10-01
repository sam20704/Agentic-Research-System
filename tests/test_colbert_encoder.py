"""Unit tests for ColBERT encoder configuration."""

from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.modules import Transformer

from src.retrieval.colbert.encoder import ColBERTEncoder


def make_transformer_stub() -> Transformer:
    """
    Create a Transformer instance without loading a real model.

    The length-configuration helper only requires a Transformer instance
    with query_length/document_length attributes, so no model download
    or CUDA device is required.
    """

    transformer = object.__new__(Transformer)
    transformer.query_length = None
    transformer.document_length = None
    return transformer


def test_apply_length_configuration_sets_query_and_document_lengths():
    transformer = make_transformer_stub()

    model = [transformer]

    ColBERTEncoder._apply_length_configuration(
        model,
        query_length=32,
        document_length=180,
    )

    assert transformer.query_length == 32
    assert transformer.document_length == 180


def test_apply_length_configuration_supports_different_lengths():
    transformer = make_transformer_stub()

    model = [transformer]

    ColBERTEncoder._apply_length_configuration(
        model,
        query_length=16,
        document_length=96,
    )

    assert transformer.query_length == 16
    assert transformer.document_length == 96


def test_apply_length_configuration_requires_exactly_one_transformer():
    model = []

    with pytest.raises(
        RuntimeError,
        match="Expected exactly one Transformer module",
    ):
        ColBERTEncoder._apply_length_configuration(
            model,
            query_length=32,
            document_length=180,
        )


def test_apply_length_configuration_rejects_multiple_transformers():
    transformer_one = make_transformer_stub()
    transformer_two = make_transformer_stub()

    model = [
        transformer_one,
        transformer_two,
    ]

    with pytest.raises(
        RuntimeError,
        match="Expected exactly one Transformer module",
    ):
        ColBERTEncoder._apply_length_configuration(
            model,
            query_length=32,
            document_length=180,
        )


def test_apply_length_configuration_does_not_require_model_download():
    transformer = make_transformer_stub()

    model = [transformer]

    ColBERTEncoder._apply_length_configuration(
        model,
        query_length=64,
        document_length=180,
    )

    assert transformer.query_length == 64
    assert transformer.document_length == 180
def test_ensure_document_tensors_preserves_variable_length_documents():
    first = torch.randn(10, 128)
    second = torch.randn(20, 128)
    third = torch.randn(7, 128)

    result = ColBERTEncoder._ensure_document_tensors(
        [first, second, third]
    )

    assert len(result) == 3
    assert result[0].shape == (10, 128)
    assert result[1].shape == (20, 128)
    assert result[2].shape == (7, 128)


def test_ensure_document_tensors_accepts_batched_tensor():
    embeddings = torch.randn(3, 12, 128)

    result = ColBERTEncoder._ensure_document_tensors(
        embeddings
    )

    assert len(result) == 3
    assert all(
        embedding.shape == (12, 128)
        for embedding in result
    )


def test_ensure_document_tensors_rejects_invalid_output_type():
    with pytest.raises(
        TypeError,
        match="Unexpected document embedding output type",
    ):
        ColBERTEncoder._ensure_document_tensors(
            "invalid"
        )


def test_ensure_document_tensors_rejects_invalid_tensor_shape():
    embeddings = torch.randn(12)

    with pytest.raises(
        ValueError,
        match="Unexpected document embedding tensor shape",
    ):
        ColBERTEncoder._ensure_document_tensors(
            embeddings
        )