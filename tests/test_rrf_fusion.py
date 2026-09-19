
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.models import DocumentChunk, RetrievalResult


def make_result(chunk_id: str, method: str):

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc",
        text="example text",
        page_numbers=(1,),
        source="policy.pdf",
    )

    return RetrievalResult(
        chunk=chunk,
        score=0.0,
        rank=1,
        retrieval_method=method,
    )


def test_duplicate_chunk_is_merged():

    bm25 = [
        make_result("chunk1", "bm25"),
        make_result("chunk2", "bm25"),
    ]

    dense = [
        make_result("chunk1", "dense"),
        make_result("chunk3", "dense"),
    ]

    fusion = ReciprocalRankFusion()

    results = fusion.fuse(bm25, dense)

    ids = [r.chunk.chunk_id for r in results]

    assert ids.count("chunk1") == 1
    assert len(ids) == 3


def test_shared_chunk_ranked_first():

    bm25 = [
        make_result("shared", "bm25"),
        make_result("bm25_only", "bm25"),
    ]

    dense = [
        make_result("shared", "dense"),
        make_result("dense_only", "dense"),
    ]

    fusion = ReciprocalRankFusion()

    results = fusion.fuse(bm25, dense)

    assert results[0].chunk.chunk_id == "shared"