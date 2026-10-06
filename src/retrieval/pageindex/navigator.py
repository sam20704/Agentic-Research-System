"""Phase 3.2 deterministic PageIndex tree navigation.

This module is intentionally independent of:
- vector embeddings
- Qdrant
- BM25
- ColBERT
- LLM inference
- agent orchestration

It takes an already-built PageIndex tree and selects the most relevant
structural nodes for a query.

The navigator is a query-time component. It does not build or modify the
PageIndex tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_TOKEN_RE = re.compile(
    r"[a-zA-Z0-9]+(?:[-/][a-zA-Z0-9]+)*|[^\W_]+",
    re.UNICODE,
)


@dataclass(frozen=True)
class PageIndexNavigationCandidate:
    """A structural node selected as a navigation candidate."""

    node: dict[str, Any]
    score: float
    rank: int
    path: tuple[str, ...]


class PageIndexNavigator:
    """Navigate an existing PageIndex tree deterministically.

    The navigator performs structural/lexical matching against node titles.

    It is deliberately kept separate from PageIndexRetriever so that the
    navigation strategy can later be replaced or augmented by:
        - a local LLM navigator,
        - PageIndex's own reasoning mechanism,
        - another structural ranking strategy,

    without changing the DocumentChunk / RetrievalResult contract.
    """

    def __init__(
        self,
        structure: dict[str, Any] | list[dict[str, Any]],
    ) -> None:
        if isinstance(structure, dict):
            nodes = structure.get("structure")
        else:
            nodes = structure

        if not isinstance(nodes, list):
            raise ValueError(
                "PageIndex structure must contain a list of nodes"
            )

        self._nodes = nodes

    def navigate(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[PageIndexNavigationCandidate]:
        """Return the highest-ranked structural nodes for a query."""

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        query_tokens = set(self._tokenize(query))

        if not query_tokens:
            return []

        candidates: list[PageIndexNavigationCandidate] = []

        for order, (node, path) in enumerate(
            self._walk(self._nodes)
        ):
            title = str(node.get("title") or "")
            title_tokens = set(self._tokenize(title))

            if not title_tokens:
                continue

            score = self._score_node(
                query=query,
                query_tokens=query_tokens,
                title=title,
                title_tokens=title_tokens,
                depth=len(path),
            )

            if score <= 0:
                continue

            candidates.append(
                PageIndexNavigationCandidate(
                    node=node,
                    score=score,
                    rank=0,
                    path=path,
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                -len(candidate.path),
                str(candidate.node.get("node_id", "")),
                candidate.path,
            )
        )

        ranked: list[PageIndexNavigationCandidate] = []

        for rank, candidate in enumerate(candidates[:top_k], start=1):
            ranked.append(
                PageIndexNavigationCandidate(
                    node=candidate.node,
                    score=candidate.score,
                    rank=rank,
                    path=candidate.path,
                )
            )

        return ranked

    @classmethod
    def _score_node(
        cls,
        *,
        query: str,
        query_tokens: set[str],
        title: str,
        title_tokens: set[str],
        depth: int,
    ) -> float:
        """Score a node using deterministic structural relevance.

        Scoring deliberately favors:
        1. exact normalized title matches,
        2. full query coverage by the title,
        3. token overlap,
        4. deeper/specific structural nodes.

        The score is a navigation score only. It is not presented as a
        probability or embedding similarity.
        """

        normalized_query = cls._normalize(query)
        normalized_title = cls._normalize(title)

        if not normalized_query or not normalized_title:
            return 0.0

        # Strongest signal: exact title equality.
        if normalized_query == normalized_title:
            return 100.0 + min(depth, 20) * 0.1

        # Strong signal: the complete query phrase occurs in the title.
        if normalized_query in normalized_title:
            return 90.0 + min(depth, 20) * 0.1

        # Strong signal in the opposite direction for short titles.
        if normalized_title in normalized_query:
            coverage = len(title_tokens) / max(len(query_tokens), 1)
            return 75.0 + (coverage * 10.0) + min(depth, 20) * 0.1

        overlap = len(query_tokens.intersection(title_tokens))

        if overlap == 0:
            return 0.0

        query_coverage = overlap / max(len(query_tokens), 1)
        title_coverage = overlap / max(len(title_tokens), 1)

        # Weighted lexical relevance.
        score = (
            (query_coverage * 50.0)
            + (title_coverage * 30.0)
            + min(depth, 20) * 0.5
        )

        return score

    @classmethod
    def _walk(
        cls,
        nodes: Iterable[dict[str, Any]],
        parent_path: tuple[str, ...] = (),
    ):
        """Yield every node together with its structural path."""

        for node in nodes:
            title = str(node.get("title") or "")
            path = parent_path + (title,)

            yield node, path

            children = node.get("nodes") or []

            if isinstance(children, list):
                yield from cls._walk(children, path)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text deterministically."""

        return [
            token.lower()
            for token in _TOKEN_RE.findall(text)
        ]

    @classmethod
    def _normalize(cls, text: str) -> str:
        """Normalize text using the deterministic tokenizer."""

        return " ".join(cls._tokenize(text))


__all__ = [
    "PageIndexNavigationCandidate",
    "PageIndexNavigator",
]