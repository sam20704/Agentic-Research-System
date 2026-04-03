from rag.loader import load_pdfs
from rag.chunker import chunk_text
from rag.embedding import embed_texts
from rag.vectorstore import store_embeddings
from rag.retriever import retrieve
from rag.cleaner import clean_text
# STEP 1: Load PDFs
docs = load_pdfs("data/references")

print(f"Loaded {len(docs)} documents")

# STEP 2: Chunk all documents
all_chunks = []
all_sources = []

for doc in docs:
    cleaned = clean_text(doc["content"])
    chunks = chunk_text(cleaned)
    all_chunks.extend(chunks)
    all_sources.extend([doc["source"]] * len(chunks))


print(f"Total chunks: {len(all_chunks)}")

chunks = [c for c in chunks if len(c.strip()) > 50]

# STEP 3: Create embeddings
embeddings = embed_texts(all_chunks)

print("Embeddings created")

# STEP 4: Store in vector DB
store_embeddings(all_chunks, embeddings, all_sources)

print("Stored in vector DB")

# STEP 5: Test retrieval
query = "What is semiconductor policy in India?"

results = retrieve(query)

print("\nTop Results:\n")

for r in results[0]:
    print("-" * 50)
    print(r[:300])