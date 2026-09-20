from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRoutingDecision:
    query: str
    is_complex: bool
    reason: str


class QueryComplexityRouter:
    """
    Lightweight deterministic query complexity router.

    Simple queries use the Phase 2 hybrid pipeline.
    Complex queries additionally invoke ColBERT.
    """

    def __init__(self, token_threshold: int = 12) -> None:
        self.token_threshold = token_threshold

    NUMERIC_PATTERN = re.compile(
        r"\b(section|page|table|figure|appendix)\s+\d+(\.\d+)*",
        re.IGNORECASE,
    )

    QUOTE_PATTERN = re.compile(r'"[^"]+"')

    COMPARISON_PATTERN = re.compile(
        r"\b(compare|versus|vs|difference)\b",
        re.IGNORECASE,
    )

    COMPLEX_KEYWORDS = {
        "eligibility",
        "criteria",
        "trusted foundry",
        "design linked incentive",
    }

    def route(self, query: str) -> QueryRoutingDecision:
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        normalized = query.lower().strip()

        tokens = normalized.split()

        # Long queries first.
        if len(tokens) >= self.token_threshold:
            return QueryRoutingDecision(
                query=query,
                is_complex=True,
                reason="query_length",
            )

        # Numeric references.
        if self.NUMERIC_PATTERN.search(normalized):
            return QueryRoutingDecision(
                query=query,
                is_complex=True,
                reason="numeric_reference",
            )

        # Quoted phrase.
        if self.QUOTE_PATTERN.search(query):
            return QueryRoutingDecision(
                query=query,
                is_complex=True,
                reason="quoted_phrase",
            )

        # Comparison queries.
        if self.COMPARISON_PATTERN.search(normalized):
            return QueryRoutingDecision(
                query=query,
                is_complex=True,
                reason="comparison",
            )

        # Domain-specific phrases.
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword in normalized:
                return QueryRoutingDecision(
                    query=query,
                    is_complex=True,
                    reason=f"keyword:{keyword}",
                )

        # Short semiconductor policy queries stay on baseline pipeline.
        return QueryRoutingDecision(
            query=query,
            is_complex=False,
            reason="simple_query",
        )