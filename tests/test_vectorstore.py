import uuid

import chromadb

from src.rag.embedding import embed_texts


def test_vectorstore_roundtrip():
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name=f"test_{uuid.uuid4().hex}"
    )

    chunks = [
        "India semiconductor policy supports domestic manufacturing.",
        "FAME II supports electric vehicle adoption.",
    ]

    embeddings = embed_texts(chunks)

    ids = [f"test_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": "test_1.pdf"},
            {"source": "test_2.pdf"},
        ],
        ids=ids,
    )

    results = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=1,
    )

    assert results["documents"]
    assert results["documents"][0]
    assert results["documents"][0][0] == chunks[0]

    client.delete_collection(collection.name)
