"""Deterministic lexical ranking for hierarchical retrieval."""

import re
from typing import Iterable, List, Tuple


_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    """Tokenize text deterministically."""
    return _TOKEN_RE.findall(text.lower())


def lexical_score(query: str, text: str, title: str = "") -> float:
    """Calculate a deterministic lexical relevance score."""
    query_tokens = _tokenize(query)
    text_tokens = _tokenize(text)
    title_tokens = set(_tokenize(title))

    if not query_tokens or not text_tokens:
        return 0.0

    text_token_set = set(text_tokens)

    overlap = sum(
        1 for token in query_tokens if token in text_token_set
    )

    score = overlap / len(query_tokens)

    normalized_query = " ".join(query_tokens)
    normalized_text = " ".join(text_tokens)

    if normalized_query and normalized_query in normalized_text:
        score += 0.25

    title_overlap = sum(
        1 for token in query_tokens if token in title_tokens
    )

    if title_overlap:
        score += 0.05 * title_overlap

    return score


def rank_items(
    query: str,
    items: Iterable[Tuple[str, str, str]],
) -> List[Tuple[str, float]]:
    """Rank ``(item_id, text, title)`` tuples deterministically."""
    scored = [
        (
            item_id,
            lexical_score(query, text, title),
        )
        for item_id, text, title in items
    ]

    scored.sort(
        key=lambda item: (-item[1], item[0])
    )

    return scored
