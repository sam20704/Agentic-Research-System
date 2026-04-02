from rag.retriever import retrieve

query = "What is semiconductor policy in India?"

results = retrieve(query)

print("\nTop Results:\n")

for r in results[0]:
    print("-" * 50)
    print(r[:300])