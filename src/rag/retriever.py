from rag.embedding import model
from rag.vectorstore import collection

def retrieve(query, top_k=3):
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"]