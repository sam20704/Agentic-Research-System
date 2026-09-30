
"""ColBERT-v2 encoder wrapper for Phase 3.1."""

from __future__ import annotations

from typing import Sequence

import torch
from sentence_transformers import MultiVectorEncoder

from src.retrieval.colbert.config import ColBERTConfig


class ColBERTEncoder:
    """
    Wrapper around Sentence Transformers MultiVectorEncoder.

    The encoder is intentionally independent from BM25, BGE-M3,
    Qdrant, RRF, reranking, and query routing.

    The underlying MultiVectorEncoder handles the ColBERT-v2
    representation, including:

    - query/document asymmetric encoding
    - token-level projection
    - token normalization
    - scoring-token masking
    - variable-length multi-vector outputs
    """

    def __init__(self, config: ColBERTConfig | None = None) -> None:
        self.config = config or ColBERTConfig()
        self.device = self._resolve_device(self.config.device)

        self.model = MultiVectorEncoder(
            self.config.model_name,
            device=str(self.device),
        )

        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve execution device."""

        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            if torch.backends.mps.is_available():
                return torch.device("mps")

            return torch.device("cpu")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "ColBERT device='cuda' was requested, "
                "but CUDA is not available."
            )

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "ColBERT device='mps' was requested, "
                "but MPS is not available."
            )

        return torch.device(device)

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def encode_query(
        self,
        query: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode one query using the checkpoint's query-side recipe.

        Returns
        -------
        tuple[Tensor, Tensor]
            Token embeddings and an all-valid scoring mask.

        MultiVectorEncoder already applies the model's scoring mask
        and returns only tokens that participate in late interaction.
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        embeddings = self.model.encode_query(
            query,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=False,
        )

        if isinstance(embeddings, list):
            if len(embeddings) != 1:
                raise RuntimeError(
                    "Expected exactly one query embedding."
                )

            token_embeddings = embeddings[0]
        else:
            token_embeddings = embeddings

        token_embeddings = token_embeddings.detach().cpu()

        mask = torch.ones(
            token_embeddings.shape[0],
            dtype=torch.bool,
        )

        return token_embeddings, mask

    # ------------------------------------------------------------------
    # Document encoding
    # ------------------------------------------------------------------

    def encode_documents(
        self,
        documents: Sequence[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Encode documents using the checkpoint's document-side recipe.

        The underlying model applies its document length and scoring
        mask. Outputs are variable-length token matrices.
        """

        if not documents:
            return [], []

        embeddings = self.model.encode_document(
            list(documents),
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
        )

        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        token_embeddings: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for embedding in embeddings:
            tensor = embedding.detach().cpu()

            token_embeddings.append(tensor)

            masks.append(
                torch.ones(
                    tensor.shape[0],
                    dtype=torch.bool,
                )
            )

        return token_embeddings, masks

    # ------------------------------------------------------------------
    # Late interaction scoring
    # ------------------------------------------------------------------

    @staticmethod
    def maxsim_score(
        query_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        document_embeddings: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> float:
        """
        Compute ColBERT MaxSim.

        For every valid query token, the maximum cosine similarity
        against document tokens is selected and the resulting values
        are summed across query tokens.
        """

        query_tokens = query_embeddings[
            query_mask.bool()
        ]

        document_tokens = document_embeddings[
            document_mask.bool()
        ]

        if query_tokens.numel() == 0:
            return 0.0

        if document_tokens.numel() == 0:
            return 0.0

        similarity = torch.matmul(
            query_tokens,
            document_tokens.T,
        )

        max_per_query = similarity.max(
            dim=1
        ).values

        return float(
            max_per_query.sum().item()
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def embedding_dimension(self) -> int:
        """Return the final ColBERT token embedding dimension."""

        # MultiVectorEncoder exposes the ColBERT projection as the
        # Dense module following the Transformer.
        #
        # For colbert-ir/colbertv2.0:
        #   Transformer: 768
        #   Dense:       768 -> 128
        #   Normalize
        #
        # Therefore the final token representation is 128-D.

        for module in self.model:
            out_features = getattr(
                module,
                "out_features",
                None,
            )

            if out_features is not None:
                return int(out_features)

        raise RuntimeError(
            "Unable to determine final ColBERT embedding dimension."
        )
