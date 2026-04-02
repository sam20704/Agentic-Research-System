# System Architecture

## Pipeline

User Query
→ Planner Agent
→ Router Agent
→ Retriever (RAG / Web)
→ Generator
→ Critic
→ Final Answer

---

## RAG Flow

Query
→ Embedding
→ Vector DB search
→ Top-k chunks
→ LLM generation

---

## Components

- Embeddings → convert text to vectors
- Vector DB → store/search chunks
- Retriever → fetch relevant chunks
- Generator → produce answer
- Critic → validate answer

---

## Failure Points

### RAG Failures
1. Bad chunking → missing context
2. Poor embeddings → wrong retrieval
3. Irrelevant chunks → hallucination

### Agent Failures
1. Wrong routing (RAG vs search)
2. Weak planning
3. No proper validation loop