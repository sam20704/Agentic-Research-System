import pytest

from src.retrieval.router import QueryComplexityRouter


def test_simple_query_routes_to_simple():
    router = QueryComplexityRouter()

    decision = router.route("India semiconductor policy")

    assert decision.is_complex is False
    assert decision.reason == "simple_query"


def test_long_query_routes_to_complex():
    router = QueryComplexityRouter()

    query = (
        "Compare semiconductor manufacturing incentives in India "
        "versus the EU Chips Act and Taiwan supply chain policies."
    )

    decision = router.route(query)

    assert decision.is_complex is True
    assert decision.reason == "query_length"


def test_numeric_query_routes_complex():
    router = QueryComplexityRouter()

    decision = router.route(
        "Section 4.3 semiconductor manufacturing incentives"
    )

    assert decision.is_complex is True
    assert "numeric_reference" in decision.reason


def test_boolean_query_routes_complex():
    router = QueryComplexityRouter()

    query = (
        "design linked incentive and manufacturing incentive "
        "eligibility under semiconductor mission"
    )

    decision = router.route(query)

    assert decision.is_complex is True


def test_quotes_route_complex():
    router = QueryComplexityRouter()

    decision = router.route(
        'Explain "trusted foundry" eligibility criteria.'
    )

    assert decision.is_complex is True
    assert "quoted_phrase" in decision.reason


def test_multiple_entities_route_complex():
    router = QueryComplexityRouter()

    query = "Compare India EU Taiwan semiconductor manufacturing policy"

    decision = router.route(query)

    assert decision.is_complex is True
    assert "comparison" in decision.reason


def test_empty_query_raises_value_error():
    router = QueryComplexityRouter()

    with pytest.raises(ValueError):
        router.route("")


def test_router_returns_decision_object():
    router = QueryComplexityRouter()

    decision = router.route("Semiconductor policy")

    assert hasattr(decision, "query")
    assert hasattr(decision, "is_complex")
    assert hasattr(decision, "reason")