"""Configuration for hierarchical retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HierarchicalRetrievalConfig:
    """Limits for progressive section, page, and chunk narrowing."""

    top_sections: int = 5
    top_pages_per_section: int = 3
    top_chunks_per_page: int = 10
    top_k: int = 10

    def __post_init__(self) -> None:
        for name in (
            "top_sections",
            "top_pages_per_section",
            "top_chunks_per_page",
            "top_k",
        ):
            value = getattr(self, name)
            # bool is an int subclass but is not a meaningful limit.
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
