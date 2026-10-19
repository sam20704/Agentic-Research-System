
"""Unit tests for Phase 3.3 hierarchical retrieval."""

import pytest

from src.retrieval.hierarchical import (
    HierarchicalRetrievalConfig,
    HierarchicalRetriever,
    HierarchyIndex,
)
from src.retrieval.models import DocumentChunk


def make_chunk(
    chunk_id: str,
    text: str,
    page: int | None = None,
    section: str | None = None,
    document_id: str = "doc-1",
    metadata: dict | None = None,
    page_numbers: tuple[int, ...] | list[int] | None = None,
) -> DocumentChunk:
    """Create a minimal DocumentChunk for testing.

    Existing callers can continue to pass a single page using ``page``.
    Tests for chunks spanning multiple pages can use ``page_numbers``.
    """
    pages = (
        list(page_numbers)
        if page_numbers is not None
        else ([page] if page is not None else [])
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        page_numbers=pages,
        source="test.pdf",
        section=section,
        bounding_boxes=[],
        element_ids=[],
        metadata=metadata or {},
    )


def test_config_defaults_are_valid():
    config = HierarchicalRetrievalConfig()

    assert config.top_sections == 5
    assert config.top_pages_per_section == 3
    assert config.top_chunks_per_page == 10
    assert config.top_k == 10


def test_config_rejects_non_positive_values():
    with pytest.raises(ValueError):
        HierarchicalRetrievalConfig(top_sections=0)

    with pytest.raises(ValueError):
        HierarchicalRetrievalConfig(top_pages_per_section=0)

    with pytest.raises(ValueError):
        HierarchicalRetrievalConfig(top_chunks_per_page=0)

    with pytest.raises(ValueError):
        HierarchicalRetrievalConfig(top_k=0)


def test_config_rejects_non_integer_values():
    with pytest.raises(TypeError):
        HierarchicalRetrievalConfig(top_sections=1.5)


def test_shallow_hierarchy_has_expected_parent_child_relationships():
    chunks = [
        make_chunk(
            "chunk-1",
            "Semiconductor supply chain policy.",
            1,
            section="Semiconductors",
        ),
        make_chunk(
            "chunk-2",
            "Electric vehicle supply chain.",
            2,
            section="Electric Vehicles",
        ),
    ]

    index = HierarchyIndex(chunks)
    sections = index.get_sections()

    assert len(sections) == 2

    semiconductor = next(
        node
        for node in sections
        if node.metadata["section_path"] == "Semiconductors"
    )

    assert semiconductor.node_type == "section"

    pages = index.get_pages(semiconductor.node_id)

    assert len(pages) == 1
    assert pages[0].node_type == "page"

    page_chunks = index.get_chunks(pages[0].node_id)

    assert len(page_chunks) == 1
    assert page_chunks[0].chunk_ids == ["chunk-1"]


def test_section_path_metadata_is_used_when_section_is_missing():
    chunks = [
        make_chunk(
            "chunk-1",
            "Regional semiconductor trends.",
            3,
            metadata={
                "section_path": "Market > Regional Trends",
            },
        )
    ]

    index = HierarchyIndex(chunks)
    sections = index.get_sections()

    assert len(sections) == 1
    assert sections[0].metadata["section_path"] == "Market"
    subsections = [
        node for node in index.nodes.values()
        if node.node_type == "subsection"
    ]
    assert len(subsections) == 1
    assert subsections[0].metadata["section_path"] == "Market > Regional Trends"
    assert subsections[0].parent_id == sections[0].node_id


def test_hierarchical_section_path_takes_precedence():
    chunks = [
        make_chunk(
            "chunk-1",
            "Supply chain analysis.",
            1,
            section="Old Section",
            metadata={
                "hierarchical_section_path": "New Section",
            },
        )
    ]

    index = HierarchyIndex(chunks)
    sections = index.get_sections()

    assert len(sections) == 1
    assert sections[0].metadata["section_path"] == "New Section"


def test_missing_headings_use_unsectioned_fallback():
    chunks = [
        make_chunk(
            "chunk-1",
            "Document text without heading.",
            1,
        )
    ]

    index = HierarchyIndex(chunks)
    sections = index.get_sections()

    assert len(sections) == 1
    assert sections[0].metadata["section_path"] == "Unsectioned"


def test_multi_page_section_contains_all_pages():
    chunks = [
        make_chunk(
            "chunk-1",
            "First page of section.",
            1,
            section="Supply Chain",
        ),
        make_chunk(
            "chunk-2",
            "Second page of section.",
            2,
            section="Supply Chain",
        ),
        make_chunk(
            "chunk-3",
            "Third page of section.",
            3,
            section="Supply Chain",
        ),
    ]

    index = HierarchyIndex(chunks)
    sections = index.get_sections()

    assert len(sections) == 1

    pages = index.get_pages(sections[0].node_id)

    assert len(pages) == 3
    assert [page.page_numbers[0] for page in pages] == [1, 2, 3]


def test_duplicate_chunk_ids_are_rejected():
    chunks = [
        make_chunk(
            "chunk-1",
            "First version.",
            1,
        ),
        make_chunk(
            "chunk-1",
            "Second version.",
            2,
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        HierarchyIndex(chunks)


def test_index_validation_succeeds_for_valid_hierarchy():
    chunks = [
        make_chunk(
            "chunk-1",
            "Valid hierarchy.",
            1,
            section="Section A",
        )
    ]

    index = HierarchyIndex(chunks)

    index.validate()


def test_multi_page_chunk_passes_hierarchy_validation():
    """A chunk may belong to more than one page without invalidating the index."""
    chunk = make_chunk(
        chunk_id="chunk-multi-page",
        text="Content spanning pages one and two.",
        page_numbers=(1, 2),
        section="Supply Chain",
    )

    index = HierarchyIndex([chunk])

    index.validate()


def test_empty_corpus_is_supported():
    index = HierarchyIndex([])

    assert index.size == 0
    assert index.get_sections() == []

    index.validate()


def test_whitespace_query_is_rejected():
    chunks = [
        make_chunk(
            "chunk-1",
            "Some content.",
            1,
        )
    ]

    retriever = HierarchicalRetriever(chunks)

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_top_k_must_be_positive():
    chunks = [
        make_chunk(
            "chunk-1",
            "Some content.",
            1,
        )
    ]

    retriever = HierarchicalRetriever(chunks)

    with pytest.raises(ValueError):
        retriever.retrieve(
            "some content",
            top_k=0,
        )


def test_retrieval_returns_existing_retrieval_result_interface():
    chunks = [
        make_chunk(
            "chunk-1",
            "Semiconductor policy and supply chain.",
            1,
            section="Semiconductors",
        ),
        make_chunk(
            "chunk-2",
            "Electric vehicle manufacturing.",
            2,
            section="Electric Vehicles",
        ),
    ]

    retriever = HierarchicalRetriever(chunks)

    results = retriever.retrieve(
        "semiconductor supply chain",
        top_k=2,
    )

    assert results
    assert results[0].retrieval_method == "hierarchical"
    assert results[0].rank == 1
    assert results[0].chunk.chunk_id == "chunk-1"


def test_retrieval_preserves_chunk_provenance():
    original = make_chunk(
        "chunk-1",
        "Semiconductor policy.",
        7,
        section="Policy",
        metadata={
            "parser": "pymupdf",
            "chunk_index": 4,
            "custom_key": "preserve-me",
        },
    )

    retriever = HierarchicalRetriever([original])

    results = retriever.retrieve(
        "semiconductor policy",
        top_k=1,
    )

    result_chunk = results[0].chunk

    assert result_chunk.chunk_id == original.chunk_id
    assert result_chunk.document_id == original.document_id
    assert result_chunk.text == original.text
    assert result_chunk.page_numbers == original.page_numbers
    assert result_chunk.source == original.source
    assert result_chunk.section == original.section
    assert result_chunk.bounding_boxes == original.bounding_boxes
    assert result_chunk.element_ids == original.element_ids

    assert result_chunk.metadata["custom_key"] == "preserve-me"
    assert result_chunk.metadata["retrieval_sources"] == ["hierarchical"]


def test_retrieval_adds_hierarchical_metadata():
    chunks = [
        make_chunk(
            "chunk-1",
            "Semiconductor policy.",
            7,
            section="Policy",
        )
    ]

    retriever = HierarchicalRetriever(chunks)

    result = retriever.retrieve(
        "semiconductor policy",
        top_k=1,
    )[0]

    metadata = result.chunk.metadata

    assert "hierarchical_score" in metadata
    assert metadata["hierarchical_rank"] == 1
    assert "hierarchical_path" in metadata
    assert metadata["hierarchical_path"] == "Policy > page 7"


def test_existing_retrieval_sources_are_preserved():
    chunks = [
        make_chunk(
            "chunk-1",
            "Semiconductor policy.",
            1,
            section="Policy",
            metadata={
                "retrieval_sources": [
                    "bm25",
                    "bge-m3",
                ]
            },
        )
    ]

    retriever = HierarchicalRetriever(chunks)

    result = retriever.retrieve(
        "semiconductor policy",
        top_k=1,
    )[0]

    assert result.chunk.metadata["retrieval_sources"] == [
        "bm25",
        "bge-m3",
        "hierarchical",
    ]


def test_retrieval_is_deterministic_for_ties():
    chunks = [
        make_chunk(
            "chunk-b",
            "same content",
            1,
            section="Section",
        ),
        make_chunk(
            "chunk-a",
            "same content",
            1,
            section="Section",
        ),
    ]

    retriever = HierarchicalRetriever(chunks)

    first = retriever.retrieve(
        "same content",
        top_k=2,
    )

    second = retriever.retrieve(
        "same content",
        top_k=2,
    )

    assert [
        result.chunk.chunk_id
        for result in first
    ] == [
        result.chunk.chunk_id
        for result in second
    ]


def test_top_k_larger_than_available_returns_all_results():
    chunks = [
        make_chunk(
            "chunk-1",
            "alpha content",
            1,
            section="Alpha",
        ),
        make_chunk(
            "chunk-2",
            "beta content",
            2,
            section="Beta",
        ),
    ]

    retriever = HierarchicalRetriever(chunks)

    results = retriever.retrieve(
        "content",
        top_k=100,
    )

    assert len(results) == 2


def test_candidate_narrowing_is_recorded():
    chunks = [
        make_chunk(
            f"chunk-{index}",
            f"content about topic {index}",
            index,
            section=f"Section {index % 3}",
        )
        for index in range(1, 21)
    ]

    retriever = HierarchicalRetriever(chunks)

    results = retriever.retrieve(
        "topic",
        top_k=5,
    )

    stats = retriever.last_stats

    assert len(results) == 5
    assert stats.corpus_chunks == 20
    assert stats.section_candidates <= 5
    assert stats.page_candidates >= 1
    assert stats.chunk_candidates >= len(results)



def test_repeated_headings_with_canonical_ids_remain_distinct():
    chunks = [
        make_chunk("overview-1", "Overview of supply chains.", 2, section="Overview", metadata={"hierarchical_section_id": "heading-2"}),
        make_chunk("overview-2", "Overview of battery logistics.", 17, section="Overview", metadata={"hierarchical_section_id": "heading-17"}),
    ]
    index = HierarchyIndex(chunks)
    sections = index.get_sections()
    assert len(sections) == 2
    assert sections[0].node_id != sections[1].node_id


def test_deep_hierarchy_preserves_all_ancestry_levels():
    chunk = make_chunk("deep-1", "Battery logistics requirements.", 4, metadata={"section_path": "Operations > Logistics > Battery Transport"})
    index = HierarchyIndex([chunk])
    sections = index.get_sections()
    assert len(sections) == 1
    level2 = index.get_children(sections[0].node_id)
    assert len(level2) == 1 and level2[0].node_type == "subsection"
    level3 = index.get_children(level2[0].node_id)
    assert len(level3) == 1 and level3[0].node_type == "subsection"
    page = index.get_pages(level3[0].node_id)[0]
    assert page.parent_id == level3[0].node_id
    assert index.get_chunks(page.node_id)[0].chunk_ids == ["deep-1"]


def test_mult_page_chunk_has_one_canonical_node_and_multiple_memberships():
    chunk = make_chunk("multi", "Evidence spans two pages.", page_numbers=(1, 2), section="Operations")
    index = HierarchyIndex([chunk])
    index.validate()
    chunk_nodes = [node for node in index.nodes.values() if node.node_type == "chunk"]
    assert len(chunk_nodes) == 1
    pages = index.get_pages(index.get_sections()[0].node_id)
    assert len(pages) == 2
    assert all([node.chunk_ids[0] for node in index.get_chunks(page.node_id)] == ["multi"] for page in pages)


def test_identical_text_with_distinct_chunk_ids_is_not_deduplicated():
    chunks = [
        make_chunk("same-1", "Identical evidence text.", 1, section="A"),
        make_chunk("same-2", "Identical evidence text.", 2, section="B"),
    ]
    results = HierarchicalRetriever(chunks).retrieve("identical evidence", top_k=5)
    assert {result.chunk.chunk_id for result in results} == {"same-1", "same-2"}

def test_retrieval_does_not_overwrite_existing_hierarchical_metadata():
    chunk = make_chunk("meta-1", "Supply chain evidence.", 1, section="Supply", metadata={"hierarchical_section_id": "source-heading-id"})
    result = HierarchicalRetriever([chunk]).retrieve("supply chain", top_k=1)[0]
    assert result.chunk.metadata["hierarchical_section_id"] == "source-heading-id"
    assert result.chunk.metadata["hierarchical_section_node_id"]
