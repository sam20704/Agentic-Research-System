import ollama

response = ollama.chat(
    model='llama3',
    messages=[
        {"role": "user", "content": "Explain RAG in 2 lines"}
    ]
)

print(response['message']['content'])