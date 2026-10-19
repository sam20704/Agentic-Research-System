"""Data models used by hierarchical retrieval."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HierarchyNode:
    """A deterministic hierarchy node.

    ``parent_id`` is reserved for structural tree nodes. Chunk nodes have no
    single parent because a canonical chunk can span multiple pages; page-to-
    chunk membership is represented by each page's ``children`` list.
    """

    node_id: str
    node_type: str
    document_id: str
    parent_id: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HierarchyStats:
    """Per-query counts that make candidate narrowing auditable.

    Existing fields retain their meaning as retained/expanded candidates.
    The additional ``*_scored`` fields expose how many nodes were actually
    evaluated at each stage.
    """

    corpus_chunks: int
    section_candidates: int
    page_candidates: int
    chunk_candidates: int
    final_results: int
    section_nodes_scored: int = 0
    page_nodes_scored: int = 0
    chunk_nodes_scored: int = 0
