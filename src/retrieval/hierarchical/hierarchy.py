"""Build and validate a deterministic hierarchy over canonical chunks."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable
from urllib.parse import quote

from src.retrieval.models import DocumentChunk

from .models import HierarchyNode


_TOKEN_RE = re.compile(r"\b\w+\b")

_SECTION_ID_KEYS = (
    "hierarchical_section_id",
    "section_id",
    "section_instance_id",
    "heading_id",
)


def _safe(value: str) -> str:
    """Encode an identifier component without losing determinism."""
    return quote(value, safe="")


class HierarchyIndex:
    """Document → section/subsection → page → chunk membership index.

    Canonical chunks are stored once. A page references a chunk through its
    ``children`` membership list, so a chunk spanning multiple pages does not
    acquire conflicting structural parents or become duplicate evidence.

    Repeated heading text can only be distinguished when ingestion supplies a
    stable structural identifier (one of ``_SECTION_ID_KEYS``). Without such
    metadata, identical paths are intentionally treated as the same section;
    inferring distinct sections from title text alone would invent structure.
    """

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self.chunks: dict[str, DocumentChunk] = {}
        self.nodes: dict[str, HierarchyNode] = {}
        self.page_chunk_ids: dict[str, list[str]] = defaultdict(list)
        self._chunk_page_node_ids: dict[str, list[str]] = defaultdict(list)

        for chunk in chunks:
            # The current retrieval contract uses chunk_id as the corpus-wide
            # identity. Reject collisions rather than silently overwrite data.
            if chunk.chunk_id in self.chunks:
                raise ValueError(f"Duplicate chunk_id detected: {chunk.chunk_id}")
            self.chunks[chunk.chunk_id] = chunk

        self._build()
        self.validate()

    @property
    def size(self) -> int:
        return len(self.chunks)

    def _build(self) -> None:
        # section key -> page -> canonical chunk IDs. All ordering is explicit.
        section_chunks: dict[tuple[str, str, str], dict[int, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        section_titles: dict[tuple[str, str, str], list[str]] = {}
        section_identities: dict[tuple[str, str, str], str | None] = {}

        for chunk in sorted(self.chunks.values(), key=lambda item: (item.document_id, item.chunk_id)):
            path = self._section_path(chunk)
            identity = self._section_identity(chunk)
            section_key = (chunk.document_id, identity or "path", path if identity is None else identity + "\x1f" + path)
            section_titles.setdefault(section_key, path.split(" > "))
            section_identities[section_key] = identity
            for page_number in sorted(set(chunk.page_numbers)):
                section_chunks[section_key][page_number].add(chunk.chunk_id)

        document_ids = sorted({chunk.document_id for chunk in self.chunks.values()})
        for document_id in document_ids:
            doc_id = self._document_node_id(document_id)
            self.nodes[doc_id] = HierarchyNode(
                node_id=doc_id, node_type="document", document_id=document_id
            )

        # Build structural nodes from each observed path. Shared prefixes are
        # shared only when their deterministic structural identity is shared.
        for section_key in sorted(section_chunks):
            document_id, _, _ = section_key
            path_parts = section_titles[section_key]
            explicit_identity = section_identities[section_key]
            parent_id = self._document_node_id(document_id)
            cumulative: list[str] = []

            for depth, title in enumerate(path_parts):
                cumulative.append(title)
                prefix = " > ".join(cumulative)
                # If a canonical structural ID is available, incorporate it
                # into the top-level identity; descendants remain path-specific.
                node_identity = explicit_identity if depth == len(path_parts) - 1 else None
                node_id = self._section_node_id(document_id, prefix, node_identity)
                node_type = "section" if depth == 0 else "subsection"
                if node_id not in self.nodes:
                    self.nodes[node_id] = HierarchyNode(
                        node_id=node_id,
                        node_type=node_type,
                        document_id=document_id,
                        parent_id=parent_id,
                        metadata={
                            "title": title,
                            "section_path": prefix,
                            "structural_identity": node_identity,
                        },
                    )
                    self.nodes[parent_id].children.append(node_id)
                parent_id = node_id

            leaf_id = parent_id
            for page_number in sorted(section_chunks[section_key]):
                page_id = self._page_node_id(document_id, leaf_id, page_number)
                if page_id not in self.nodes:
                    self.nodes[page_id] = HierarchyNode(
                        node_id=page_id,
                        node_type="page",
                        document_id=document_id,
                        parent_id=leaf_id,
                        page_numbers=[page_number],
                    )
                    self.nodes[leaf_id].children.append(page_id)
                for chunk_id in sorted(section_chunks[section_key][page_number]):
                    if chunk_id not in self.page_chunk_ids[page_id]:
                        self.page_chunk_ids[page_id].append(chunk_id)
                        self._chunk_page_node_ids[chunk_id].append(page_id)

        # One canonical chunk node per chunk, independent of page membership.
        for chunk in sorted(self.chunks.values(), key=lambda item: item.chunk_id):
            node_id = self._chunk_node_id(chunk.chunk_id)
            self.nodes[node_id] = HierarchyNode(
                node_id=node_id,
                node_type="chunk",
                document_id=chunk.document_id,
                page_numbers=list(chunk.page_numbers),
                chunk_ids=[chunk.chunk_id],
                text=chunk.text,
                metadata=dict(chunk.metadata),
            )

        # Make each page's chunk membership visible in the graph, but do not
        # set chunk.parent_id: a chunk may belong to multiple page nodes.
        for page_id, chunk_ids in self.page_chunk_ids.items():
            self.nodes[page_id].children.extend(
                self._chunk_node_id(chunk_id) for chunk_id in sorted(chunk_ids)
            )

        # Precompute compact, deterministic summaries once at index-build time.
        # Keep unique tokens in first-seen order instead of copying every full
        # descendant chunk into every ancestor node. Full text remains only on
        # the canonical chunk objects and is scored after page selection.
        for node in self.nodes.values():
            if node.node_type in {"section", "subsection", "page"}:
                descendant_ids = self._descendant_chunk_ids(node.node_id)
                seen_tokens: set[str] = set()
                summary_tokens: list[str] = []
                for chunk_id in descendant_ids:
                    for token in _TOKEN_RE.findall(self.chunks[chunk_id].text.lower()):
                        if token not in seen_tokens:
                            seen_tokens.add(token)
                            summary_tokens.append(token)
                node.text = " ".join(summary_tokens)

        for node in self.nodes.values():
            node.children.sort()

    @staticmethod
    def _section_path(chunk: DocumentChunk) -> str:
        metadata = chunk.metadata or {}
        for key in ("hierarchical_section_path", "section_path"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return " > ".join(part.strip() for part in value.split(">") if part.strip()) or "Unsectioned"
        if chunk.section and chunk.section.strip():
            return " > ".join(part.strip() for part in chunk.section.split(">") if part.strip()) or "Unsectioned"
        return "Unsectioned"

    @staticmethod
    def _section_identity(chunk: DocumentChunk) -> str | None:
        metadata = chunk.metadata or {}
        for key in _SECTION_ID_KEYS:
            value = metadata.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _document_node_id(document_id: str) -> str:
        return f"document::{_safe(document_id)}"

    @staticmethod
    def _section_node_id(document_id: str, path: str, identity: str | None) -> str:
        suffix = f"::identity::{_safe(identity)}" if identity is not None else ""
        return f"section::{_safe(document_id)}::{_safe(path)}{suffix}"

    @staticmethod
    def _page_node_id(document_id: str, parent_id: str, page_number: int) -> str:
        return f"page::{_safe(document_id)}::{_safe(parent_id)}::{page_number}"

    @staticmethod
    def _chunk_node_id(chunk_id: str) -> str:
        return f"chunk::{_safe(chunk_id)}"

    def _descendant_chunk_ids(self, node_id: str) -> list[str]:
        node = self.nodes[node_id]
        found: set[str] = set()
        stack = list(reversed(node.children))
        while stack:
            child_id = stack.pop()
            child = self.nodes.get(child_id)
            if child is None:
                continue
            if child.node_type == "chunk":
                found.update(child.chunk_ids)
            else:
                stack.extend(reversed(child.children))
        return sorted(found)

    def get_node(self, node_id: str) -> HierarchyNode | None:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[HierarchyNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_sections(self) -> list[HierarchyNode]:
        """Return top-level section nodes (compatibility API)."""
        return sorted((n for n in self.nodes.values() if n.node_type == "section"), key=lambda n: n.node_id)

    def get_section_candidates(self) -> list[HierarchyNode]:
        """Return section and subsection nodes eligible for first-stage ranking."""
        return sorted(
            (n for n in self.nodes.values() if n.node_type in {"section", "subsection"}),
            key=lambda n: (n.metadata.get("section_path", ""), n.node_id),
        )

    def get_pages(self, section_id: str) -> list[HierarchyNode]:
        """Return descendant pages for a section/subsection, not chunk nodes."""
        node = self.nodes.get(section_id)
        if node is None:
            return []
        pages: dict[str, HierarchyNode] = {}
        stack = list(reversed(node.children))
        while stack:
            child_id = stack.pop()
            child = self.nodes.get(child_id)
            if child is None:
                continue
            if child.node_type == "page":
                pages[child.node_id] = child
            elif child.node_type in {"section", "subsection"}:
                stack.extend(reversed(child.children))
        return sorted(pages.values(), key=lambda n: (n.page_numbers[0] if n.page_numbers else 0, n.node_id))

    def get_chunks(self, page_id: str) -> list[HierarchyNode]:
        return [self.nodes[self._chunk_node_id(cid)] for cid in sorted(self.page_chunk_ids.get(page_id, []))]

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        return self.chunks.get(chunk_id)

    def validate(self) -> None:
        """Reject broken structural links, memberships, and cycles."""
        for node_id, node in self.nodes.items():
            if node.node_id != node_id:
                raise ValueError(f"Node key/id mismatch for {node_id!r}")
            if node.parent_id is not None:
                parent = self.nodes.get(node.parent_id)
                if parent is None:
                    raise ValueError(f"Missing parent {node.parent_id!r} for node {node_id!r}")
                if node_id not in parent.children:
                    raise ValueError(f"Parent {node.parent_id!r} does not reference child {node_id!r}")
            seen_children: set[str] = set()
            for child_id in node.children:
                if child_id in seen_children:
                    raise ValueError(f"Duplicate child reference {child_id!r} in {node_id!r}")
                seen_children.add(child_id)
                child = self.nodes.get(child_id)
                if child is None:
                    raise ValueError(f"Missing child {child_id!r} for node {node_id!r}")
                if child.node_type == "chunk":
                    if node.node_type != "page" or child.chunk_ids[0] not in self.page_chunk_ids.get(node_id, []):
                        raise ValueError(f"Invalid chunk membership {child_id!r} under {node_id!r}")
                elif child.parent_id != node_id:
                    raise ValueError(f"Invalid parent reference for {child_id!r}")
        for page_id, chunk_ids in self.page_chunk_ids.items():
            if page_id not in self.nodes or self.nodes[page_id].node_type != "page":
                raise ValueError(f"Invalid page membership owner {page_id!r}")
            for chunk_id in chunk_ids:
                if chunk_id not in self.chunks:
                    raise ValueError(f"Page {page_id!r} references missing chunk {chunk_id!r}")
        for chunk_id, chunk in self.chunks.items():
            if chunk_id != chunk.chunk_id:
                raise ValueError(f"Chunk key/id mismatch for {chunk_id!r}")
            if not self._chunk_page_node_ids.get(chunk_id):
                raise ValueError(f"Orphan chunk {chunk_id!r} has no page membership")
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()
        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError(f"Cycle detected at node {node_id!r}")
            if node_id in visited:
                return
            visiting.add(node_id)
            node = self.nodes[node_id]
            # Chunk memberships are edges for validation, but chunk nodes are
            # leaves and therefore cannot form cycles in a valid construction.
            for child_id in node.children:
                visit(child_id)
            visiting.remove(node_id)
            visited.add(node_id)
        for node_id in sorted(self.nodes):
            visit(node_id)
