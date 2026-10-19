"""Deterministic hierarchical candidate narrowing over canonical chunks."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from src.retrieval.models import DocumentChunk, RetrievalResult

from .config import HierarchicalRetrievalConfig
from .hierarchy import HierarchyIndex
from .models import HierarchyNode, HierarchyStats
from .ranking import lexical_score


class HierarchicalRetriever:
    """Narrow candidates section → page → chunk, then rank final evidence.

    Section and page text summaries are built once by ``HierarchyIndex``.
    Query-time section ranking therefore does not recursively rescan all
    descendant chunks; page expansion happens only after section selection,
    and chunk scoring happens only after page selection.
    """

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        config: HierarchicalRetrievalConfig | None = None,
    ) -> None:
        self.config = config or HierarchicalRetrievalConfig()
        self.index = HierarchyIndex(chunks)
        self.last_stats = HierarchyStats(
            corpus_chunks=self.index.size,
            section_candidates=0,
            page_candidates=0,
            chunk_candidates=0,
            final_results=0,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        requested_top_k = self.config.top_k if top_k is None else top_k
        if isinstance(requested_top_k, bool) or not isinstance(requested_top_k, int):
            raise TypeError("top_k must be an integer")
        if requested_top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        all_sections = self._rank_sections(query)
        selected_sections = all_sections[: self.config.top_sections]

        selected_pages_by_id: dict[str, tuple[HierarchyNode, float, float]] = {}
        page_nodes_scored = 0
        for section, section_score in selected_sections:
            ranked_pages = self._rank_pages(query, section)
            page_nodes_scored += len(ranked_pages)
            for page, page_score in ranked_pages[: self.config.top_pages_per_section]:
                combined = 0.4 * section_score + 0.6 * page_score
                previous_page = selected_pages_by_id.get(page.node_id)
                candidate_page = (page, combined, section_score)
                if previous_page is None or combined > previous_page[1]:
                    selected_pages_by_id[page.node_id] = candidate_page

        selected_pages = sorted(
            selected_pages_by_id.values(),
            key=lambda item: (-item[1], item[0].node_id),
        )

        # Canonical chunks are deduplicated by chunk_id, not by text. Keep the
        # best route score if one chunk is reachable from more than one page.
        candidates: dict[str, tuple[DocumentChunk, float, str]] = {}
        chunk_nodes_scored = 0
        for page, page_score, _ in selected_pages:
            ranked_chunks = self._rank_chunks(query, page)
            chunk_nodes_scored += len(ranked_chunks)
            for chunk_node, chunk_score in ranked_chunks[: self.config.top_chunks_per_page]:
                chunk_id = chunk_node.chunk_ids[0]
                chunk = self.index.chunks[chunk_id]
                score = 0.35 * page_score + 0.65 * chunk_score
                previous = candidates.get(chunk_id)
                candidate = (chunk, score, page.node_id)
                if previous is None or score > previous[1] or (
                    score == previous[1] and page.node_id < previous[2]
                ):
                    candidates[chunk_id] = candidate

        ranked_candidates = sorted(
            candidates.values(),
            key=lambda item: (-item[1], item[0].document_id, item[0].primary_page, item[0].chunk_id),
        )
        final_candidates = ranked_candidates[:requested_top_k]

        results: list[RetrievalResult] = []
        for rank, (chunk, score, page_node_id) in enumerate(final_candidates, start=1):
            metadata = dict(chunk.metadata)
            raw_sources = metadata.get("retrieval_sources", [])
            if isinstance(raw_sources, str):
                sources = [raw_sources]
            elif isinstance(raw_sources, (list, tuple)):
                sources = [str(item) for item in raw_sources]
            else:
                sources = []
            if "hierarchical" not in sources:
                sources.append("hierarchical")

            page_node = self.index.get_node(page_node_id)
            path, section_id, subsection_id = self._hierarchical_path(page_node)
            metadata.update({
                "retrieval_sources": sources,
                "hierarchical_score": score,
                "hierarchical_rank": rank,
                "hierarchical_path": path,
                "hierarchical_section_node_id": section_id,
                "hierarchical_page_node_id": page_node_id,
            })
            if subsection_id is not None:
                metadata.setdefault("hierarchical_subsection_id", subsection_id)

            result_chunk = replace(chunk, metadata=metadata)
            results.append(RetrievalResult(
                chunk=result_chunk,
                score=score,
                rank=rank,
                retrieval_method="hierarchical",
                metadata={
                    "hierarchical_path": path,
                    "hierarchical_score": score,
                    "hierarchical_section_node_id": section_id,
                    "hierarchical_page_node_id": page_node_id,
                },
            ))

        self.last_stats = HierarchyStats(
            corpus_chunks=self.index.size,
            section_candidates=len(selected_sections),
            page_candidates=len(selected_pages),
            chunk_candidates=len(candidates),
            final_results=len(results),
            section_nodes_scored=len(all_sections),
            page_nodes_scored=page_nodes_scored,
            chunk_nodes_scored=chunk_nodes_scored,
        )
        return results

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        return self.retrieve(query, top_k=top_k)

    def _rank_sections(self, query: str) -> list[tuple[HierarchyNode, float]]:
        ranked = []
        for section in self.index.get_section_candidates():
            title = str(section.metadata.get("title", section.metadata.get("section_path", "")))
            ranked.append((section, lexical_score(query, section.text, title)))
        return sorted(ranked, key=lambda item: (-item[1], item[0].node_id))

    def _rank_pages(self, query: str, section: HierarchyNode) -> list[tuple[HierarchyNode, float]]:
        ranked = []
        for page in self.index.get_pages(section.node_id):
            ranked.append((page, lexical_score(query, page.text)))
        return sorted(ranked, key=lambda item: (-item[1], item[0].node_id))

    def _rank_chunks(self, query: str, page: HierarchyNode) -> list[tuple[HierarchyNode, float]]:
        ranked = []
        for chunk_node in self.index.get_chunks(page.node_id):
            ranked.append((chunk_node, lexical_score(query, chunk_node.text)))
        return sorted(ranked, key=lambda item: (-item[1], item[0].node_id))

    def _hierarchical_path(self, page_node: HierarchyNode | None) -> tuple[str, str | None, str | None]:
        if page_node is None:
            return "", None, None
        parts: list[str] = []
        section_id: str | None = None
        subsection_id: str | None = None
        current = self.index.get_node(page_node.parent_id or "")
        while current is not None and current.node_type != "document":
            if current.node_type == "section":
                section_id = current.node_id
            elif current.node_type == "subsection":
                subsection_id = current.node_id
            title = current.metadata.get("title")
            if isinstance(title, str) and title:
                parts.append(title)
            current = self.index.get_node(current.parent_id or "")
        parts.reverse()
        page_number = page_node.page_numbers[0] if page_node.page_numbers else None
        path = " > ".join(parts + ([f"page {page_number}"] if page_number is not None else []))
        return path, section_id, subsection_id
