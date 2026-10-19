"""Integration tests for Phase 3.3 hierarchical retrieval."""

from src.retrieval.hierarchical import (
    HierarchicalRetrievalConfig,
    HierarchicalRetriever,
)
from src.retrieval.models import DocumentChunk


def make_chunk(
    chunk_id: str,
    text: str,
    page: int,
    section: str | None = None,
    document_id: str = "doc-1",
    metadata: dict | None = None,
) -> DocumentChunk:
    """Create a minimal DocumentChunk for integration testing."""
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        page_numbers=[page],
        source="integration-test.pdf",
        section=section,
        bounding_boxes=[],
        element_ids=[],
        metadata=metadata or {},
    )


def test_hierarchical_retrieval_narrows_candidates_across_levels():
    chunks = [
        make_chunk(
            "sem-1",
            "Semiconductor manufacturing policy and chip supply.",
            1,
            section="Semiconductors",
        ),
        make_chunk(
            "sem-2",
            "Advanced semiconductor fabrication facilities.",
            2,
            section="Semiconductors",
        ),
        make_chunk(
            "sem-3",
            "Semiconductor export controls and technology policy.",
            3,
            section="Semiconductors",
        ),
        make_chunk(
            "ev-1",
            "Electric vehicle manufacturing and battery demand.",
            4,
            section="Electric Vehicles",
        ),
        make_chunk(
            "ev-2",
            "EV charging infrastructure development.",
            5,
            section="Electric Vehicles",
        ),
        make_chunk(
            "supply-1",
            "Global logistics and shipping capacity.",
            6,
            section="Global Supply Chain",
        ),
    ]

    config = HierarchicalRetrievalConfig(
        top_sections=1,
        top_pages_per_section=2,
        top_chunks_per_page=2,
        top_k=3,
    )

    retriever = HierarchicalRetriever(
        chunks,
        config=config,
    )

    results = retriever.retrieve(
        "semiconductor manufacturing policy",
        top_k=3,
    )

    assert results
    assert all(
        result.chunk.section == "Semiconductors"
        for result in results
    )

    stats = retriever.last_stats

    assert stats.corpus_chunks == 6
    assert stats.section_candidates == 1
    assert stats.page_candidates <= 2
    assert stats.chunk_candidates <= 3
    assert stats.final_results <= 3


def test_hierarchical_retrieval_preserves_multiple_documents():
    chunks = [
        make_chunk(
            "doc1-1",
            "Semiconductor policy in the first document.",
            1,
            section="Policy",
            document_id="doc-1",
        ),
        make_chunk(
            "doc2-1",
            "Semiconductor policy in the second document.",
            1,
            section="Policy",
            document_id="doc-2",
        ),
    ]

    retriever = HierarchicalRetriever(chunks)

    results = retriever.retrieve(
        "semiconductor policy",
        top_k=10,
    )

    result_ids = [
        result.chunk.chunk_id
        for result in results
    ]

    assert result_ids == [
        "doc1-1",
        "doc2-1",
    ]


def test_hierarchical_retrieval_preserves_existing_metadata_and_provenance():
    chunk = make_chunk(
        "chunk-1",
        "Semiconductor supply chain policy.",
        12,
        section="Semiconductors > Policy",
        metadata={
            "parser": "pymupdf",
            "chunk_index": 8,
            "element_type": "text",
            "retrieval_sources": ["bm25", "bge-m3"],
            "custom_provenance": {
                "document": "source.pdf",
                "page": 12,
            },
        },
    )

    retriever = HierarchicalRetriever([chunk])

    results = retriever.retrieve(
        "semiconductor supply chain policy",
        top_k=1,
    )

    assert len(results) == 1

    result_chunk = results[0].chunk

    assert result_chunk.chunk_id == chunk.chunk_id
    assert result_chunk.document_id == chunk.document_id
    assert result_chunk.source == chunk.source
    assert result_chunk.page_numbers == chunk.page_numbers
    assert result_chunk.section == chunk.section
    assert result_chunk.bounding_boxes == chunk.bounding_boxes
    assert result_chunk.element_ids == chunk.element_ids

    assert result_chunk.metadata["parser"] == "pymupdf"
    assert result_chunk.metadata["chunk_index"] == 8
    assert result_chunk.metadata["element_type"] == "text"
    assert result_chunk.metadata["custom_provenance"] == {
        "document": "source.pdf",
        "page": 12,
    }

    assert result_chunk.metadata["retrieval_sources"] == [
        "bm25",
        "bge-m3",
        "hierarchical",
    ]


def test_hierarchical_retrieval_result_interface_matches_existing_contract():
    chunks = [
        make_chunk(
            "chunk-1",
            "Semiconductor manufacturing capacity.",
            1,
            section="Semiconductors",
        ),
        make_chunk(
            "chunk-2",
            "Battery manufacturing capacity.",
            2,
            section="Electric Vehicles",
        ),
    ]

    retriever = HierarchicalRetriever(chunks)

    results = retriever.retrieve(
        "semiconductor manufacturing",
        top_k=2,
    )

    assert results

    result = results[0]

    assert hasattr(result, "chunk")
    assert hasattr(result, "score")
    assert hasattr(result, "rank")
    assert hasattr(result, "retrieval_method")
    assert hasattr(result, "metadata")

    assert result.retrieval_method == "hierarchical"
    assert result.rank == 1
    assert isinstance(result.score, float)
    assert isinstance(result.metadata, dict)


def test_hierarchical_retrieval_is_deterministic():
    chunks = [
        make_chunk(
            "chunk-c",
            "Supply chain policy.",
            3,
            section="Policy",
        ),
        make_chunk(
            "chunk-a",
            "Supply chain policy.",
            1,
            section="Policy",
        ),
        make_chunk(
            "chunk-b",
            "Supply chain policy.",
            2,
            section="Policy",
        ),
    ]

    config = HierarchicalRetrievalConfig(
        top_sections=5,
        top_pages_per_section=5,
        top_chunks_per_page=5,
        top_k=3,
    )

    retriever = HierarchicalRetriever(
        chunks,
        config=config,
    )

    first = retriever.retrieve(
        "supply chain policy",
        top_k=3,
    )

    second = retriever.retrieve(
        "supply chain policy",
        top_k=3,
    )

    assert [
        (
            result.chunk.chunk_id,
            result.score,
            result.rank,
        )
        for result in first
    ] == [
        (
            result.chunk.chunk_id,
            result.score,
            result.rank,
        )
        for result in second
    ]

def test_repeated_execution_has_identical_ids_scores_and_provenance():
    chunks = [
        make_chunk("b", "Semiconductor supply chain policy.", 2, section="Policy"),
        make_chunk("a", "Semiconductor supply chain policy.", 1, section="Policy"),
        make_chunk("c", "Battery transport rules.", 3, section="Transport"),
    ]
    retriever = HierarchicalRetriever(chunks, HierarchicalRetrievalConfig(top_sections=2, top_pages_per_section=3, top_chunks_per_page=3, top_k=3))
    first = retriever.retrieve("semiconductor supply chain")
    first_snapshot = [(r.chunk.chunk_id, r.score, r.rank, r.chunk.page_numbers, r.chunk.metadata) for r in first]
    second = retriever.retrieve("semiconductor supply chain")
    second_snapshot = [(r.chunk.chunk_id, r.score, r.rank, r.chunk.page_numbers, r.chunk.metadata) for r in second]
    assert first_snapshot == second_snapshot


def test_candidate_scoring_counts_are_observable():
    chunks = [
        make_chunk("a1", "Semiconductor policy details.", 1, section="Semiconductors"),
        make_chunk("a2", "Semiconductor trade details.", 2, section="Semiconductors"),
        make_chunk("b1", "Battery charging details.", 3, section="Batteries"),
    ]
    retriever = HierarchicalRetriever(chunks, HierarchicalRetrievalConfig(top_sections=1, top_pages_per_section=1, top_chunks_per_page=1, top_k=1))
    retriever.retrieve("semiconductor policy")
    stats = retriever.last_stats
    assert stats.section_nodes_scored == 2
    assert stats.section_candidates == 1
    assert stats.page_nodes_scored == 2
    assert stats.page_candidates == 1
    assert stats.chunk_nodes_scored == 1
    assert stats.chunk_candidates == 1
    assert stats.final_results == 1
