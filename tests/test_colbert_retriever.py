import pytest
import torch

from src.retrieval.colbert.config import ColBERTConfig
from src.retrieval.colbert.index import ColBERTIndex
from src.retrieval.colbert.retriever import ColBERTRetriever
from src.retrieval.models import DocumentChunk


# ------------------------------------------------------------------
# Fake deterministic encoder (no model download)
# ------------------------------------------------------------------

class FakeColBERTEncoder:
    def __init__(self):
        self.config = ColBERTConfig(device="cpu")

    def encode_documents(self, texts):
        embeddings = []
        masks = []

        for text in texts:
            if "semiconductor" in text.lower():
                emb = torch.tensor([[1.0, 0.0], [0.8, 0.0]])
            elif "vehicle" in text.lower():
                emb = torch.tensor([[0.0, 1.0], [0.0, 0.8]])
            else:
                emb = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

            embeddings.append(emb)
            masks.append(torch.tensor([1, 1]))

        return embeddings, masks

    def encode_query(self, query):
        if "semiconductor" in query.lower():
            emb = torch.tensor([[1.0, 0.0]])
        elif "vehicle" in query.lower():
            emb = torch.tensor([[0.0, 1.0]])
        else:
            emb = torch.tensor([[0.5, 0.5]])

        return emb, torch.tensor([1])

    def maxsim_score(
        self,
        query_embeddings,
        query_mask,
        document_embeddings,
        document_mask,
    ):
        scores = torch.matmul(query_embeddings, document_embeddings.T)
        return scores.max(dim=1).values.sum()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_chunk(chunk_id, text, page):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        page_numbers=(page,),
        source="policy.pdf",
        metadata={"section": "Policy"},
    )


def make_retriever():
    encoder = FakeColBERTEncoder()
    index = ColBERTIndex(encoder)

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


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------

def test_build_index_counts_chunks():
    retriever = make_retriever()
    assert retriever.index_size() == 3


def test_semiconductor_query_ranks_correct_chunk_first():
    retriever = make_retriever()

    results = retriever.retrieve("semiconductor policy")

    assert results[0].chunk.chunk_id == "chunk1"
    assert results[0].retrieval_method == "colbert"


def test_vehicle_query_ranks_correct_chunk_first():
    retriever = make_retriever()

    results = retriever.retrieve("electric vehicle incentives")

    assert results[0].chunk.chunk_id == "chunk2"


def test_top_k_limits_results():
    retriever = make_retriever()

    results = retriever.retrieve("policy", top_k=2)

    assert len(results) == 2


def test_empty_index_returns_empty_results():
    encoder = FakeColBERTEncoder()
    retriever = ColBERTRetriever(
        config=encoder.config,
        encoder=encoder,
        index=ColBERTIndex(encoder),
    )

    assert retriever.retrieve("semiconductor") == []


def test_empty_query_raises_value_error():
    retriever = make_retriever()

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_provenance_is_preserved():
    retriever = make_retriever()

    result = retriever.retrieve("semiconductor")[0]

    assert result.chunk.document_id == "doc1"
    assert result.chunk.page_numbers == (1,)
    assert result.chunk.source == "policy.pdf"
    assert result.chunk.metadata["section"] == "Policy"
    assert result.chunk.metadata["retrieval_sources"] == ["colbert"]
    assert "colbert_score" in result.chunk.metadata


def test_deterministic_ranking():
    retriever = make_retriever()

    first = retriever.retrieve("semiconductor")
    second = retriever.retrieve("semiconductor")

    assert [r.chunk.chunk_id for r in first] == [
        r.chunk.chunk_id for r in second
    ]