import chromadb
from chromadb.config import Settings
import uuid
client = chromadb.Client(Settings(persist_directory="./chroma_db"))

collection = client.get_or_create_collection(name="docs")

def store_embeddings(chunks, embeddings, sources):
    import uuid
    
    for chunk, emb, src in zip(chunks, embeddings, sources):
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            metadatas=[{"source": src}],
            ids=[str(uuid.uuid4())]
        )