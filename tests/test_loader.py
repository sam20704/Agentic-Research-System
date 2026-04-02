from rag.loader import load_pdfs
from rag.chunker import chunk_text

docs = load_pdfs("data/references")

chunks = chunk_text(docs[0]["content"])

print(f"Total chunks: {len(chunks)}")
print("\nSample chunk:\n")
print(chunks[0])