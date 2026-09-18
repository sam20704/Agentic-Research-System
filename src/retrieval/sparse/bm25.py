"""BM25 sparse retrieval."""

from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from src.retrieval.models import DocumentChunk, RetrievalResult


class BM25Retriever:
    """Lexical BM25 retriever over DocumentChunk objects."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b

        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None

        if chunks is not None:
            self.index(chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text while preserving useful lexical terms."""

        return re.findall(
            r"[a-zA-Z0-9]+(?:[-/][a-zA-Z0-9]+)*|₹|[^\W_]+",
            text.lower(),
            flags=re.UNICODE,
        )

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Replace the current index with the supplied chunks."""

        if not chunks:
            raise ValueError("chunks must not be empty")

        chunk_list = list(chunks)

        if any(not chunk.text.strip() for chunk in chunk_list):
            raise ValueError("chunks must contain non-empty text")

        tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in chunk_list
        ]

        if any(not tokens for tokens in tokenized_chunks):
            raise ValueError("chunks must contain tokenizable text")

        self._chunks = chunk_list
        self._bm25 = BM25Okapi(
            tokenized_chunks,
            k1=self.k1,
            b=self.b,
        )

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Add chunks to the existing index."""

        if not chunks:
            raise ValueError("chunks must not be empty")

        if not self._chunks:
            self.index(chunks)
            return

        self.index([*self._chunks, *chunks])

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the highest-scoring lexical matches."""

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (-scores[index], index),
        )

        results: list[RetrievalResult] = []

        for index in ranked_indices:
            score = float(scores[index])

            # Return only documents containing at least one query token.
            # A valid lexical match may have a BM25 score of 0.0 when
            # the query term occurs in every indexed document.
            document_tokens = self._bm25.doc_freqs[index]

            if not any(
                token in document_tokens
                for token in query_tokens
            ):
                continue

            results.append(
                RetrievalResult(
                    chunk=self._chunks[index],
                    score=score,
                    rank=len(results) + 1,
                    retrieval_method="bm25",
                )
            )

            if len(results) >= top_k:
                break

        return results

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return len(self._chunks)

    def clear(self) -> None:
        """Remove all indexed chunks."""

        self._chunks = []
        self._bm25 = None