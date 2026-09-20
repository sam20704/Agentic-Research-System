import pytest

from src.retrieval.router import QueryComplexityRouter


def test_simple_query_is_simple():
    router = QueryComplexityRouter()

    decision = router.route("India semiconductor policy")

    assert decision.is_complex is False
    assert decision.reason == "simple_query"


def test_long_query_is_complex():
    router = QueryComplexityRouter(token_threshold=6)

    decision = router.route(
        "Compare semiconductor manufacturing incentives across India EU Taiwan"
    )

    assert decision.is_complex is True
    assert decision.reason == "query_length"


def test_keyword_query_is_complex():
    router = QueryComplexityRouter()

    decision = router.route(
        'Explain "trusted foundry" eligibility criteria'
    )

    assert decision.is_complex is True


def test_empty_query_raises():
    router = QueryComplexityRouter()

    with pytest.raises(ValueError):
        router.route("")