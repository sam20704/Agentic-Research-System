import os

import pytest
import ollama


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1",
    reason="LLM integration tests disabled by default",
)
def test_llm_smoke():
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": "Explain RAG in 2 lines"}
        ],
    )

    content = response["message"]["content"].strip()

    assert content
