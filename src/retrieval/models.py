"""Core data models for the retrieval layer."""

from dataclasses import dataclass, field
from typing import Any

from src.document.models import BoundingBox


@dataclass(frozen=True)
class DocumentChunk:
    """A retrieval unit derived from a canonical document.

    A chunk keeps enough provenance information to trace retrieved
    evidence back to the original document and page locations.
    """

    chunk_id: str
    document_id: str
    text: str

    page_numbers: tuple[int, ...]
    source: str

    section: str | None = None

    bounding_boxes: tuple[BoundingBox, ...] = ()
    element_ids: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")

        if not self.document_id:
            raise ValueError("document_id must not be empty")

        if not self.text.strip():
            raise ValueError("chunk text must not be empty")

        if not self.page_numbers:
            raise ValueError("chunk must reference at least one page")

        if any(page < 1 for page in self.page_numbers):
            raise ValueError("page numbers must be >= 1")

        if len(self.bounding_boxes) != len(self.element_ids):
            raise ValueError(
                "bounding_boxes and element_ids must have the same length"
            )

    @property
    def primary_page(self) -> int:
        """Return the first page associated with the chunk."""
        return self.page_numbers[0]

    @property
    def page_count(self) -> int:
        """Return the number of pages represented by the chunk."""
        return len(self.page_numbers)


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk together with its retrieval metadata."""

    chunk: DocumentChunk
    score: float
    rank: int
    retrieval_method: str

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")

        if not self.retrieval_method:
            raise ValueError("retrieval_method must not be empty")