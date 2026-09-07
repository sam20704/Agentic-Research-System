from src.rag.retriever import retrieve


def test_retriever_returns_results():
    query = "What is semiconductor policy in India?"

    results = retrieve(query)

    assert isinstance(results, list)
    assert results

    for result in results:
        assert isinstance(result, str)
        assert result.strip()
