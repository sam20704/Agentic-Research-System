from rag.embedding import embed_texts

embeddings = embed_texts(chunks)

print(f"Embedding shape: {len(embeddings)}")