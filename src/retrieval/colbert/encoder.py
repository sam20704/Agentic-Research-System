from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from src.retrieval.colbert.config import ColBERTConfig


class ColBERTEncoder:
    """
    Wrapper around the ColBERT model.

    Separates query encoding and document encoding while sharing
    the underlying tokenizer/model instance.
    """

    def __init__(self, config: ColBERTConfig | None = None) -> None:
        self.config = config or ColBERTConfig()

        self.device = self._resolve_device(self.config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        ).to(self.device)

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

        return torch.device(device)

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def encode_query(
        self,
        query: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single query.

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor)
            (token_embeddings, attention_mask)
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        encoded = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_query_length,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = self.model(**encoded)

        token_embeddings = outputs.last_hidden_state.squeeze(0).cpu()
        attention_mask = encoded["attention_mask"].squeeze(0).cpu()

        return token_embeddings, attention_mask

    # ------------------------------------------------------------------
    # Document encoding
    # ------------------------------------------------------------------

    def encode_documents(
        self,
        documents: Sequence[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Encode multiple documents.

        Returns
        -------
        tuple(list[Tensor], list[Tensor])
            (document_embeddings, attention_masks)

        Each list contains one entry per document.
        """

        if not documents:
            return [], []

        embeddings: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for start in range(
            0,
            len(documents),
            self.config.batch_size,
        ):
            batch = documents[start : start + self.config.batch_size]

            encoded = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_document_length,
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with torch.inference_mode():
                outputs = self.model(**encoded)

            hidden_states = outputs.last_hidden_state.cpu()
            attention_masks = encoded["attention_mask"].cpu()

            # One embedding tensor + one mask per document.
            for hidden, mask in zip(
                hidden_states,
                attention_masks,
                strict=True,
            ):
                embeddings.append(hidden)
                masks.append(mask)

        return embeddings, masks

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
        ColBERT MaxSim score.

        Computes maximum similarity between each valid query token
        and all valid document tokens, then sums across query tokens.
        """

        query_tokens = query_embeddings[query_mask.bool()]
        document_tokens = document_embeddings[document_mask.bool()]

        similarity = torch.matmul(
            query_tokens,
            document_tokens.T,
        )

        max_per_query = similarity.max(dim=1).values

        return float(max_per_query.sum().item())

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def embedding_dimension(self) -> int:
        """Return token embedding dimension."""

        return self.model.config.hidden_size