from src.rag.embedding import model
from src.rag.vectorstore import collection


def retrieve(query, top_k=3, verbose=False):
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if verbose:
        print("Querying vector DB...")
        print(f"Collection size: {collection.count()}")

    documents = results.get("documents", [])
    if not documents or not documents[0]:
        return []

    return documents[0]