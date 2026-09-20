"""Qwen3 cross-encoder reranking for hybrid retrieval results."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.retrieval.models import RetrievalResult


DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 8

DEFAULT_INSTRUCTION = (
    "Given a research query, retrieve passages that directly answer "
    "the query and provide relevant factual evidence."
)

_SYSTEM_PROMPT = "Judge whether the Document is relevant to the Query."

_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    f"{_SYSTEM_PROMPT}\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
)

_PROMPT_SUFFIX = (
    "\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class CrossEncoderReranker:
    """Rerank retrieval candidates using Qwen3-Reranker-0.6B."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        instruction: str = DEFAULT_INSTRUCTION,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be non-empty")

        if max_length < 1:
            raise ValueError("max_length must be >= 1")

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if not instruction or not instruction.strip():
            raise ValueError("instruction must be non-empty")

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")

        self.device = torch.device(device)

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )

        self.model = model or (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(
                    torch.float16
                    if self.device.type == "cuda"
                    else torch.float32
                ),
            )
            .to(self.device)
            .eval()
        )

        self._yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self._no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        if self._yes_token_id is None or self._no_token_id is None:
            raise ValueError(
                "Qwen reranker requires 'yes' and 'no' tokenizer tokens"
            )

    def _build_prompt(self, query: str, document: str) -> str:
        return (
            _PROMPT_PREFIX
            + f"<Instruct>: {self.instruction}\n"
            + f"<Query>: {query}\n"
            + f"<Document>: {document}"
            + _PROMPT_SUFFIX
        )

    def _score_documents(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        prompts = [
            self._build_prompt(query, document)
            for document in documents
        ]

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits[:, -1, :]

            relevance_logits = logits[
                :,
                [self._no_token_id, self._yes_token_id],
            ]

            scores = torch.softmax(relevance_logits, dim=-1)[:, 1]

        return scores.float().cpu().tolist()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Rerank candidates for a query and return the top results."""

        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if not candidates:
            return []

        documents = [candidate.chunk.text for candidate in candidates]
        scores: list[float] = []

        for start in range(0, len(documents), self.batch_size):
            batch_documents = documents[start : start + self.batch_size]
            scores.extend(self._score_documents(query, batch_documents))

        scored_candidates = []

        for candidate, score in zip(candidates, scores):
            metadata = dict(candidate.metadata)

            # Preserve the original fusion score.
            metadata["rrf_score"] = candidate.score

            # Store the Qwen reranker score explicitly.
            metadata["reranker_score"] = score

            scored_candidates.append(
                (
                    candidate.chunk,
                    score,
                    metadata,
                )
            )

        scored_candidates.sort(
            key=lambda item: (-item[1], item[0].chunk_id)
        )

        results = []

        for rank, (chunk, score, metadata) in enumerate(
            scored_candidates[:top_k],
            start=1,
        ):
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_method="cross_encoder",
                    metadata=metadata,
                )
            )

        return results