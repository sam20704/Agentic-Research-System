from rag.retriever import retrieve
from rag.generator import generate_answer

query = "What incentives are provided in India's semiconductor policy?"

# STEP 1: Retrieve relevant chunks
results = retrieve(query)

contexts = retrieve(query)

# STEP 2: Generate answer
answer = generate_answer(query, contexts)

print("\nANSWER:\n")
print(answer)
print("\nRETRIEVED CONTEXTS:\n")

for i, ctx in enumerate(contexts):
    print(f"\n--- Context {i+1} ---\n")
    print(ctx[:300])