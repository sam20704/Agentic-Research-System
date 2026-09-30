"""Unit tests for the independent Phase 3.1 ColBERT retriever."""

import pytest
import torch

from src.document.models import BoundingBox
from src.retrieval.colbert.config import ColBERTConfig
from src.retrieval.colbert.index import ColBERTIndex
from src.retrieval.colbert.retriever import ColBERTRetriever
from src.retrieval.models import DocumentChunk


# ---------------------------------------------------------------------
# Fake deterministic encoder
# ---------------------------------------------------------------------


class FakeColBERTEncoder:
    """Deterministic encoder used without model downloads."""

    def __init__(self) -> None:
        self.config = ColBERTConfig(device="cpu")

    def encode_documents(self, texts):
        embeddings = []
        masks = []

        for text in texts:
            lowered = text.lower()

            if "semiconductor" in lowered:
                embedding = torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.8, 0.0],
                    ]
                )
            elif "vehicle" in lowered:
                embedding = torch.tensor(
                    [
                        [0.0, 1.0],
                        [0.0, 0.8],
                    ]
                )
            else:
                embedding = torch.tensor(
                    [
                        [0.5, 0.5],
                        [0.5, 0.5],
                    ]
                )

            embeddings.append(embedding)
            masks.append(
                torch.tensor(
                    [True, True],
                    dtype=torch.bool,
                )
            )

        return embeddings, masks

    def encode_query(self, query):
        lowered = query.lower()

        if "semiconductor" in lowered:
            embedding = torch.tensor(
                [[1.0, 0.0]]
            )
        elif "vehicle" in lowered:
            embedding = torch.tensor(
                [[0.0, 1.0]]
            )
        else:
            embedding = torch.tensor(
                [[0.5, 0.5]]
            )

        return (
            embedding,
            torch.tensor(
                [True],
                dtype=torch.bool,
            ),
        )

    @staticmethod
    def maxsim_score(
        query_embeddings,
        query_mask,
        document_embeddings,
        document_mask,
    ):
        query_tokens = query_embeddings[
            query_mask.bool()
        ]

        document_tokens = document_embeddings[
            document_mask.bool()
        ]

        scores = torch.matmul(
            query_tokens,
            document_tokens.T,
        )

        return scores.max(
            dim=1
        ).values.sum()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_chunk(
    chunk_id: str,
    text: str,
    page: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        page_numbers=(page,),
        source="policy.pdf",
        section="Policy",
        bounding_boxes=(
            BoundingBox(
                x0=1,
                y0=2,
                x1=3,
                y1=4,
            ),
        ),
        element_ids=(f"{chunk_id}-element",),
        metadata={"source_label": "fixture"},
    )


def make_retriever() -> ColBERTRetriever:
    encoder = FakeColBERTEncoder()

    index = ColBERTIndex(
        encoder=encoder,
        config=encoder.config,
    )

    retriever = ColBERTRetriever(
        config=encoder.config,
        encoder=encoder,
        index=index,
    )

    retriever.build_index(
        [
            make_chunk(
                "chunk1",
                "India semiconductor policy incentives.",
                1,
            ),
            make_chunk(
                "chunk2",
                "Electric vehicle adoption incentives.",
                2,
            ),
            make_chunk(
                "chunk3",
                "Generic manufacturing report.",
                5,
            ),
        ]
    )

    return retriever


# ---------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------


def test_build_index_counts_chunks():
    retriever = make_retriever()

    assert retriever.index_size() == 3


def test_semiconductor_query_ranks_correct_chunk_first():
    retriever = make_retriever()

    results = retriever.retrieve(
        "semiconductor policy"
    )

    assert results[0].chunk.chunk_id == "chunk1"
    assert results[0].retrieval_method == "colbert"


def test_vehicle_query_ranks_correct_chunk_first():
    retriever = make_retriever()

    results = retriever.retrieve(
        "electric vehicle incentives"
    )

    assert results[0].chunk.chunk_id == "chunk2"


def test_top_k_limits_results():
    retriever = make_retriever()

    results = retriever.retrieve(
        "policy",
        top_k=2,
    )

    assert len(results) == 2


def test_empty_index_returns_empty_results():
    encoder = FakeColBERTEncoder()

    retriever = ColBERTRetriever(
        config=encoder.config,
        encoder=encoder,
        index=ColBERTIndex(
            encoder=encoder,
            config=encoder.config,
        ),
    )

    assert retriever.retrieve(
        "semiconductor"
    ) == []


def test_empty_query_raises_value_error():
    retriever = make_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_non_positive_top_k_raises_value_error():
    retriever = make_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve(
            "semiconductor",
            top_k=0,
        )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "semiconductor",
            top_k=-1,
        )


def test_provenance_is_preserved():
    retriever = make_retriever()

    result = retriever.retrieve(
        "semiconductor"
    )[0]

    assert result.chunk.chunk_id == "chunk1"
    assert result.chunk.document_id == "doc1"
    assert result.chunk.page_numbers == (1,)
    assert result.chunk.source == "policy.pdf"
    assert result.chunk.section == "Policy"
    assert result.chunk.bounding_boxes == (
        BoundingBox(
            x0=1,
            y0=2,
            x1=3,
            y1=4,
        ),
    )
    assert result.chunk.element_ids == (
        "chunk1-element",
    )

    assert result.chunk.metadata[
        "source_label"
    ] == "fixture"

    assert result.chunk.metadata[
        "retrieval_sources"
    ] == ["colbert"]

    assert "colbert_score" in result.chunk.metadata


def test_deterministic_ranking():
    retriever = make_retriever()

    first = retriever.retrieve(
        "semiconductor"
    )

    second = retriever.retrieve(
        "semiconductor"
    )

    assert [
        result.chunk.chunk_id
        for result in first
    ] == [
        result.chunk.chunk_id
        for result in second
    ]


def test_index_build_is_deterministic_regardless_of_input_order():
    encoder_a = FakeColBERTEncoder()
    encoder_b = FakeColBERTEncoder()

    retriever_a = ColBERTRetriever(
        config=encoder_a.config,
        encoder=encoder_a,
        index=ColBERTIndex(
            encoder=encoder_a,
            config=encoder_a.config,
        ),
    )

    retriever_b = ColBERTRetriever(
        config=encoder_b.config,
        encoder=encoder_b,
        index=ColBERTIndex(
            encoder=encoder_b,
            config=encoder_b.config,
        ),
    )

    chunks = [
        make_chunk(
            "chunk2",
            "Electric vehicle adoption incentives.",
            2,
        ),
        make_chunk(
            "chunk1",
            "India semiconductor policy incentives.",
            1,
        ),
        make_chunk(
            "chunk3",
            "Generic manufacturing report.",
            5,
        ),
    ]

    retriever_a.build_index(chunks)
    retriever_b.build_index(list(reversed(chunks)))

    first = retriever_a.retrieve(
        "semiconductor"
    )

    second = retriever_b.retrieve(
        "semiconductor"
    )

    assert [
        result.chunk.chunk_id
        for result in first
    ] == [
        result.chunk.chunk_id
        for result in second
    ]


def test_duplicate_chunk_ids_are_rejected():
    encoder = FakeColBERTEncoder()

    retriever = ColBERTRetriever(
        config=encoder.config,
        encoder=encoder,
        index=ColBERTIndex(
            encoder=encoder,
            config=encoder.config,
        ),
    )

    duplicate_chunks = [
        make_chunk(
            "chunk1",
            "India semiconductor policy.",
            1,
        ),
        make_chunk(
            "chunk1",
            "Another document chunk.",
            2,
        ),
    ]

    with pytest.raises(ValueError):
        retriever.build_index(
            duplicate_chunks
        )


def test_top_k_equal_one_returns_one_result():
    retriever = make_retriever()

    results = retriever.retrieve(
        "semiconductor",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1


def test_top_k_equal_index_size_returns_all_results():
    retriever = make_retriever()

    results = retriever.retrieve(
        "semiconductor",
        top_k=3,
    )

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]


def test_top_k_larger_than_index_returns_all_results():
    retriever = make_retriever()

    results = retriever.retrieve(
        "semiconductor",
        top_k=100,
    )

    assert len(results) == retriever.index_size()
    assert [result.rank for result in results] == [1, 2, 3]


def test_empty_chunk_list_clears_index():
    retriever = make_retriever()

    retriever.build_index([])

    assert retriever.index_size() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": ""},
        {"model_name": "   "},
        {"batch_size": 0},
        {"batch_size": -1},
        {"max_query_length": 0},
        {"max_query_length": -1},
        {"max_document_length": 0},
        {"max_document_length": -1},
        {"default_top_k": 0},
        {"default_top_k": -1},
        {"device": "invalid-device"},
    ],
)
def test_invalid_colbert_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ColBERTConfig(**kwargs)
