from src.rag.embedding import embed_texts


def test_embeddings():
    chunks = [
        "India has introduced policies to support semiconductor manufacturing.",
        "Electric vehicles are supported through government incentives.",
    ]

    embeddings = embed_texts(chunks)

    assert len(embeddings) == len(chunks)
    assert len(embeddings[0]) > 0
