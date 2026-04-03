import ollama

def generate_answer(query, contexts):
    if not contexts or len(contexts) == 0:
        return "I don't have enough information to answer this."

    contexts = contexts[:5]

    context_text = "\n\n".join(
        [f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    prompt = f"""
You are a strict retrieval-grounded QA assistant.

Answer the question using ONLY the provided context blocks.
Do not use prior knowledge.
Do not speculate.
Do not fill gaps.

If you include any information not present in the context, the answer is incorrect.

If the answer is not explicitly stated in the context, reply exactly:
"I don't have enough information to answer this."

Instructions:
- Every answer must be directly supported by one or more context blocks.
- If support is weak, partial, or ambiguous, do not answer.
- Keep the answer short and factual.

Format:

Answer:
<your answer>

Support:
[Context X], [Context Y]

Context:
{context_text}

Question:
{query}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]