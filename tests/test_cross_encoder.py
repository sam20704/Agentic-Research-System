import pytest
import torch

from src.document.models import BoundingBox
from src.retrieval.models import DocumentChunk, RetrievalResult
from src.retrieval.reranking.cross_encoder import CrossEncoderReranker


class FakeTokenizer:
    def __init__(self):
        self.last_prompts = []

    def convert_tokens_to_ids(self, token):
        return {"yes": 1, "no": 0}[token]

    def __call__(
        self,
        prompts,
        padding=True,
        truncation=True,
        max_length=None,
        return_tensors="pt",
    ):
        self.last_prompts = prompts

        # One fake token per prompt.
        return FakeBatch(len(prompts))


class FakeBatch(dict):
    def __init__(self, batch_size):
        super().__init__()
        self["input_ids"] = torch.ones((batch_size, 1), dtype=torch.long)
        self["attention_mask"] = torch.ones((batch_size, 1), dtype=torch.long)

    def to(self, device):
        return self


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]

        start = self.calls
        batch_scores = self.scores[start : start + batch_size]
        self.calls += batch_size

        logits = torch.zeros(
            (batch_size, 1, 2),
            dtype=torch.float32,
        )

        for index, score in enumerate(batch_scores):
            # yes token = 1, no token = 0.
            logits[index, 0, 1] = score
            logits[index, 0, 0] = 0.0

        return FakeOutput(logits)


def make_result(
    chunk_id,
    text,
    score,
    *,
    page=1,
):
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        text=text,
        page_numbers=(page,),
        source="test.pdf",
        section="Section A",
        bounding_boxes=(
            BoundingBox(
                x0=10.0,
                y0=20.0,
                x1=100.0,
                y1=120.0,
            ),
        ),
        element_ids=(f"element-{chunk_id}",),
        metadata={"parser": "test"},
    )

    return RetrievalResult(
        chunk=chunk,
        score=score,
        rank=1,
        retrieval_method="hybrid",
        metadata={
            "rrf_score": score,
            "bm25_rank": 1,
            "dense_rank": 2,
            "retrieval_sources": ["bm25", "dense"],
        },
    )


def make_reranker(scores, batch_size=8):
    tokenizer = FakeTokenizer()
    model = FakeModel(scores)

    reranker = CrossEncoderReranker(
        device="cpu",
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
    )

    return reranker, tokenizer, model


def test_empty_candidates_returns_empty():
    reranker, _, _ = make_reranker([])

    assert reranker.rerank("test query", []) == []


def test_single_candidate():
    reranker, _, _ = make_reranker([4.0])

    candidate = make_result(
        "chunk-1",
        "Relevant document",
        0.03,
    )

    results = reranker.rerank(
        "test query",
        [candidate],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].retrieval_method == "cross_encoder"
    assert results[0].score > 0.5


def test_reranking_changes_order():
    reranker, _, _ = make_reranker([1.0, 5.0, 2.0])

    candidates = [
        make_result("chunk-1", "first", 0.05),
        make_result("chunk-2", "second", 0.04),
        make_result("chunk-3", "third", 0.03),
    ]

    results = reranker.rerank(
        "test query",
        candidates,
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]

    assert [result.rank for result in results] == [1, 2, 3]


def test_top_k_limits_results():
    reranker, _, _ = make_reranker([1.0, 5.0, 2.0])

    candidates = [
        make_result("chunk-1", "first", 0.05),
        make_result("chunk-2", "second", 0.04),
        make_result("chunk-3", "third", 0.03),
    ]

    results = reranker.rerank(
        "test query",
        candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert [result.chunk.chunk_id for result in results] == [
        "chunk-2",
        "chunk-3",
    ]


def test_rrf_score_is_preserved():
    reranker, _, _ = make_reranker([5.0])

    candidate = make_result(
        "chunk-1",
        "document",
        0.03125,
    )

    result = reranker.rerank(
        "query",
        [candidate],
    )[0]

    assert result.metadata["rrf_score"] == pytest.approx(0.03125)
    assert result.metadata["reranker_score"] == pytest.approx(
        result.score
    )


def test_original_retrieval_metadata_is_preserved():
    reranker, _, _ = make_reranker([5.0])

    candidate = make_result(
        "chunk-1",
        "document",
        0.03,
    )

    result = reranker.rerank(
        "query",
        [candidate],
    )[0]

    assert result.metadata["bm25_rank"] == 1
    assert result.metadata["dense_rank"] == 2
    assert result.metadata["retrieval_sources"] == [
        "bm25",
        "dense",
    ]


def test_chunk_provenance_is_preserved():
    reranker, _, _ = make_reranker([5.0])

    candidate = make_result(
        "chunk-1",
        "document",
        0.03,
        page=7,
    )

    result = reranker.rerank(
        "query",
        [candidate],
    )[0]

    assert result.chunk.chunk_id == "chunk-1"
    assert result.chunk.document_id == "doc-001"
    assert result.chunk.page_numbers == (7,)
    assert result.chunk.source == "test.pdf"
    assert result.chunk.section == "Section A"
    assert result.chunk.element_ids == ("element-chunk-1",)
    assert len(result.chunk.bounding_boxes) == 1
    assert result.chunk.metadata["parser"] == "test"


def test_prompt_contains_query_instruction_and_document():
    reranker, tokenizer, _ = make_reranker([5.0])

    candidate = make_result(
        "chunk-1",
        "India semiconductor incentives",
        0.03,
    )

    reranker.rerank(
        "What incentives does India provide?",
        [candidate],
    )

    prompt = tokenizer.last_prompts[0]

    assert "What incentives does India provide?" in prompt
    assert "India semiconductor incentives" in prompt
    assert "Given a research query" in prompt
    assert "<Query>:" in prompt
    assert "<Document>:" in prompt


def test_candidates_are_processed_in_batches():
    reranker, _, model = make_reranker(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        batch_size=2,
    )

    candidates = [
        make_result(f"chunk-{index}", f"document {index}", 0.01)
        for index in range(5)
    ]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=5,
    )

    assert len(results) == 5
    assert model.calls == 5


@pytest.mark.parametrize(
    "query, candidates, top_k, message",
    [
        ("", [], 1, "query"),
        ("   ", [], 1, "query"),
    ],
)
def test_invalid_query(query, candidates, top_k, message):
    reranker, _, _ = make_reranker([])

    with pytest.raises(ValueError, match=message):
        reranker.rerank(query, candidates, top_k)


def test_invalid_top_k():
    reranker, _, _ = make_reranker([])

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("query", [], top_k=0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_length": 0}, "max_length"),
        ({"batch_size": 0}, "batch_size"),
        ({"instruction": ""}, "instruction"),
    ],
)
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CrossEncoderReranker(
            device="cpu",
            tokenizer=FakeTokenizer(),
            model=FakeModel([]),
            **kwargs,
        )