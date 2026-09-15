"""Tests for the Phase 2 Qdrant vector store."""

from __future__ import annotations

import math

import pytest
from qdrant_client import QdrantClient

from src.document.models import BoundingBox
from src.retrieval.models import DocumentChunk
from src.retrieval.vectorstore import QdrantVectorStore


VECTOR_SIZE = 4


def make_chunk(
    chunk_id: str = "doc-001:chunk:0:test",
    document_id: str = "doc-001",
    page_number: int = 1,
) -> DocumentChunk:
    """Create a representative DocumentChunk for testing."""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text="India provides incentives for semiconductor manufacturing.",
        page_numbers=(page_number,),
        source="semiconductor_policy.pdf",
        section="Semiconductor Policy",
        bounding_boxes=(
            BoundingBox(
                x0=10.0,
                y0=20.0,
                x1=300.0,
                y1=80.0,
            ),
        ),
        element_ids=("element-001",),
        metadata={
            "parser": "pymupdf",
            "element_type": "paragraph",
        },
    )


@pytest.fixture
def store() -> QdrantVectorStore:
    """Create an isolated in-memory Qdrant store for each test."""

    client = QdrantClient(":memory:")

    return QdrantVectorStore(
        collection_name="test_chunks",
        vector_size=VECTOR_SIZE,
        client=client,
    )


def test_collection_is_created(store: QdrantVectorStore) -> None:
    """The vector store should create its collection automatically."""

    assert store.client.collection_exists("test_chunks")
    assert store.collection_name == "test_chunks"
    assert store.count() == 0


def test_upsert_and_count(store: QdrantVectorStore) -> None:
    """Chunks and embeddings should be stored successfully."""

    chunks = [
        make_chunk(
            chunk_id="doc-001:chunk:0:test",
            page_number=1,
        ),
        make_chunk(
            chunk_id="doc-001:chunk:1:test",
            page_number=2,
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    stored = store.upsert(chunks, embeddings)

    assert stored == 2
    assert store.count() == 2


def test_get_returns_original_chunk(
    store: QdrantVectorStore,
) -> None:
    """Stored chunks should be reconstructable from Qdrant."""

    chunk = make_chunk()
    embedding = [[1.0, 0.0, 0.0, 0.0]]

    store.upsert([chunk], embedding)

    results = store.get([chunk.chunk_id])

    assert len(results) == 1

    retrieved = results[0]

    assert retrieved.chunk_id == chunk.chunk_id
    assert retrieved.document_id == chunk.document_id
    assert retrieved.text == chunk.text
    assert retrieved.page_numbers == chunk.page_numbers
    assert retrieved.source == chunk.source
    assert retrieved.section == chunk.section
    assert retrieved.element_ids == chunk.element_ids
    assert retrieved.metadata == chunk.metadata


def test_provenance_survives_round_trip(
    store: QdrantVectorStore,
) -> None:
    """Page and bounding-box provenance should survive storage."""

    chunk = make_chunk()

    store.upsert(
        [chunk],
        [[1.0, 0.0, 0.0, 0.0]],
    )

    retrieved = store.get([chunk.chunk_id])[0]

    assert retrieved.page_numbers == (1,)
    assert retrieved.element_ids == ("element-001",)

    assert len(retrieved.bounding_boxes) == 1

    bbox = retrieved.bounding_boxes[0]

    assert bbox.x0 == 10.0
    assert bbox.y0 == 20.0
    assert bbox.x1 == 300.0
    assert bbox.y1 == 80.0


def test_get_unknown_chunk_returns_empty(
    store: QdrantVectorStore,
) -> None:
    """Unknown chunk IDs should not produce fabricated results."""

    results = store.get(["does-not-exist"])

    assert results == []


def test_upsert_rejects_mismatched_lengths(
    store: QdrantVectorStore,
) -> None:
    """The number of chunks must match the number of embeddings."""

    chunks = [
        make_chunk(
            chunk_id="doc-001:chunk:0:test",
        ),
        make_chunk(
            chunk_id="doc-001:chunk:1:test",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
    ]

    with pytest.raises(ValueError, match="same length"):
        store.upsert(chunks, embeddings)


def test_upsert_rejects_wrong_vector_dimension(
    store: QdrantVectorStore,
) -> None:
    """Embeddings must match the configured vector dimension."""

    chunk = make_chunk()

    with pytest.raises(ValueError, match="dimension"):
        store.upsert(
            [chunk],
            [[1.0, 0.0, 0.0]],
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_upsert_rejects_non_finite_embeddings(
    store: QdrantVectorStore,
    invalid_value: float,
) -> None:
    """Embeddings containing NaN or infinity should be rejected."""

    chunk = make_chunk()

    embedding = [
        invalid_value,
        0.0,
        0.0,
        0.0,
    ]

    with pytest.raises(ValueError, match="finite"):
        store.upsert([chunk], [embedding])


def test_upsert_replaces_existing_chunk(
    store: QdrantVectorStore,
) -> None:
    """Upserting the same chunk ID should update the existing point."""

    chunk = make_chunk()

    store.upsert(
        [chunk],
        [[1.0, 0.0, 0.0, 0.0]],
    )

    updated_chunk = DocumentChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text="Updated semiconductor policy text.",
        page_numbers=chunk.page_numbers,
        source=chunk.source,
        section=chunk.section,
        bounding_boxes=chunk.bounding_boxes,
        element_ids=chunk.element_ids,
        metadata={
            "parser": "docling",
            "element_type": "paragraph",
        },
    )

    store.upsert(
        [updated_chunk],
        [[0.0, 1.0, 0.0, 0.0]],
    )

    assert store.count() == 1

    retrieved = store.get([chunk.chunk_id])[0]

    assert retrieved.text == "Updated semiconductor policy text."
    assert retrieved.metadata["parser"] == "docling"


def test_delete_document(
    store: QdrantVectorStore,
) -> None:
    """All chunks belonging to a document should be deletable."""

    chunks = [
        make_chunk(
            chunk_id="doc-001:chunk:0:test",
            document_id="doc-001",
            page_number=1,
        ),
        make_chunk(
            chunk_id="doc-001:chunk:1:test",
            document_id="doc-001",
            page_number=2,
        ),
        make_chunk(
            chunk_id="doc-002:chunk:0:test",
            document_id="doc-002",
            page_number=1,
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    store.upsert(chunks, embeddings)

    assert store.count() == 3

    store.delete_document("doc-001")

    assert store.count() == 1

    remaining = store.get(
        ["doc-002:chunk:0:test"]
    )

    assert len(remaining) == 1
    assert remaining[0].document_id == "doc-002"


def test_delete_document_rejects_empty_id(
    store: QdrantVectorStore,
) -> None:
    """An empty document ID should be rejected."""

    with pytest.raises(
        ValueError,
        match="document_id",
    ):
        store.delete_document("")


def test_clear_removes_all_chunks(
    store: QdrantVectorStore,
) -> None:
    """Clearing the store should recreate an empty collection."""

    chunk = make_chunk()

    store.upsert(
        [chunk],
        [[1.0, 0.0, 0.0, 0.0]],
    )

    assert store.count() == 1

    store.clear()

    assert store.client.collection_exists(
        store.collection_name
    )
    assert store.count() == 0


def test_empty_upsert_is_noop(
    store: QdrantVectorStore,
) -> None:
    """Upserting no chunks should safely do nothing."""

    stored = store.upsert([], [])

    assert stored == 0
    assert store.count() == 0


def test_empty_get_is_noop(
    store: QdrantVectorStore,
) -> None:
    """Getting no chunk IDs should return an empty list."""

    assert store.get([]) == []
    