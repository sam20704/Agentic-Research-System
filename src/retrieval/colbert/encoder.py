"""ColBERT encoder and late-interaction scoring utilities."""

from __future__ import annotations

from typing import Sequence

import torch
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.sentence_transformer.modules import Transformer

from src.retrieval.colbert.config import ColBERTConfig


class ColBERTEncoder:
    """Encode queries/documents and compute ColBERT MaxSim scores."""

    def __init__(self, config: ColBERTConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)

        self.model = MultiVectorEncoder(
            self.config.model_name,
            device=str(self.device),
        )
        self.model.eval()

        self._apply_length_configuration(
            self.model,
            query_length=config.max_query_length,
            document_length=config.max_document_length,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve and validate the configured execution device."""

        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")

            return torch.device("cpu")

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "ColBERT device 'cuda' was requested, "
                    "but CUDA is unavailable."
                )

            return torch.device("cuda")

        if device == "mps":
            if (
                not hasattr(torch.backends, "mps")
                or not torch.backends.mps.is_available()
            ):
                raise RuntimeError(
                    "ColBERT device 'mps' was requested, "
                    "but MPS is unavailable."
                )

            return torch.device("mps")

        if device == "cpu":
            return torch.device("cpu")

        raise ValueError(
            f"Unsupported ColBERT device '{device}'. "
            "Choose from 'auto', 'cpu', 'cuda', or 'mps'."
        )

    @staticmethod
    def _apply_length_configuration(
        model: MultiVectorEncoder,
        *,
        query_length: int,
        document_length: int,
    ) -> None:
        """
        Apply query/document lengths to the model's Transformer module.
        """

        transformer_modules = [
            module
            for module in model
            if isinstance(module, Transformer)
        ]

        if len(transformer_modules) != 1:
            raise RuntimeError(
                "Expected exactly one Transformer module in the ColBERT model; "
                f"found {len(transformer_modules)}."
            )

        transformer = transformer_modules[0]
        transformer.query_length = query_length
        transformer.document_length = document_length

    @staticmethod
    def _ensure_tensor(embeddings: object) -> torch.Tensor:
        """Convert a single model embedding output to a tensor."""

        if isinstance(embeddings, torch.Tensor):
            return embeddings

        return torch.as_tensor(embeddings)

    @staticmethod
    def _ensure_document_tensors(
        embeddings: object,
    ) -> list[torch.Tensor]:
        """
        Normalize real ColBERT document output to a list of tensors.

        MultiVectorEncoder.encode_document() returns one tensor per
        document because documents can have different token counts.
        """

        if isinstance(embeddings, torch.Tensor):
            if embeddings.ndim == 2:
                return [embeddings]

            if embeddings.ndim == 3:
                return [
                    embedding
                    for embedding in embeddings
                ]

            raise ValueError(
                "Unexpected document embedding tensor shape: "
                f"{tuple(embeddings.shape)}."
            )

        if not isinstance(embeddings, (list, tuple)):
            raise TypeError(
                "Unexpected document embedding output type: "
                f"{type(embeddings).__name__}."
            )

        document_embeddings: list[torch.Tensor] = []

        for embedding in embeddings:
            tensor = (
                embedding
                if isinstance(embedding, torch.Tensor)
                else torch.as_tensor(embedding)
            )

            if tensor.ndim != 2:
                raise ValueError(
                    "Each document embedding must have shape "
                    "[tokens, embedding_dim]; "
                    f"found {tuple(tensor.shape)}."
                )

            document_embeddings.append(tensor)

        return document_embeddings

    def encode_query(
        self,
        query: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single query.

        Returns:
            A tuple containing token embeddings and a boolean attention mask.
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        with torch.inference_mode():
            embeddings = self.model.encode_query(
                query,
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=False,
            )

        embeddings = self._ensure_tensor(embeddings)

        if embeddings.ndim == 3:
            if embeddings.shape[0] != 1:
                raise ValueError(
                    "Expected a single query embedding or a batch "
                    "containing exactly one query."
                )

            embeddings = embeddings[0]

        if embeddings.ndim != 2:
            raise ValueError(
                "Expected query embeddings with shape "
                "[tokens, embedding_dim]; "
                f"found {tuple(embeddings.shape)}."
            )

        attention_mask = torch.ones(
            embeddings.shape[0],
            dtype=torch.bool,
            device=embeddings.device,
        )

        return embeddings, attention_mask

    def encode_documents(
        self,
        documents: Sequence[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Encode a batch of documents.

        Returns:
            A list of token-embedding tensors and a corresponding list
            of boolean attention masks.
        """

        if not documents:
            raise ValueError("documents must not be empty.")

        if any(
            not document or not document.strip()
            for document in documents
        ):
            raise ValueError(
                "documents must not contain empty text."
            )

        with torch.inference_mode():
            embeddings = self.model.encode_document(
                list(documents),
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=False,
            )

        document_embeddings = self._ensure_document_tensors(
            embeddings
        )

        if len(document_embeddings) != len(documents):
            raise RuntimeError(
                "ColBERT returned a different number of document "
                "embeddings than input documents: "
                f"expected {len(documents)}, "
                f"found {len(document_embeddings)}."
            )

        attention_masks = [
            torch.ones(
                embedding.shape[0],
                dtype=torch.bool,
                device=embedding.device,
            )
            for embedding in document_embeddings
        ]

        return document_embeddings, attention_masks

    @staticmethod
    def maxsim_score(
        query_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        document_embeddings: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ColBERT MaxSim.

        Args:
            query_embeddings:
                Query token embeddings with shape
                [query_tokens, embedding_dim] or
                [batch, query_tokens, embedding_dim].

            query_mask:
                Boolean query-token mask.

            document_embeddings:
                Document token embeddings with shape
                [document_tokens, embedding_dim] or
                [batch, document_tokens, embedding_dim].

            document_mask:
                Boolean document-token mask.

        Returns:
            MaxSim score for each query/document pair.
        """

        if query_embeddings.ndim == 2:
            query_embeddings = query_embeddings.unsqueeze(0)

        if document_embeddings.ndim == 2:
            document_embeddings = document_embeddings.unsqueeze(0)

        if query_mask.ndim == 1:
            query_mask = query_mask.unsqueeze(0)

        if document_mask.ndim == 1:
            document_mask = document_mask.unsqueeze(0)

        similarity = torch.matmul(
            query_embeddings,
            document_embeddings.transpose(-1, -2),
        )

        document_mask_expanded = document_mask.unsqueeze(1)

        similarity = similarity.masked_fill(
            ~document_mask_expanded,
            torch.finfo(similarity.dtype).min,
        )

        max_similarity = similarity.max(dim=-1).values

        query_mask_float = query_mask.to(
            dtype=max_similarity.dtype
        )

        scores = (
            max_similarity * query_mask_float
        ).sum(dim=-1)

        return scores

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of ColBERT token embeddings."""

        for module in self.model:
            out_features = getattr(
                module,
                "out_features",
                None,
            )

            if out_features is not None:
                return int(out_features)

        raise RuntimeError(
            "Unable to determine ColBERT embedding dimension "
            "from model modules."
        )