from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(chunks):
    embeddings = model.encode(chunks)
    return embeddings