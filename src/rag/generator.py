import time
import ollama

REFUSAL_TEXT = "I don't have enough information to answer this."


def generate_answer(query, contexts):
    if not contexts:
        return REFUSAL_TEXT

    contexts = contexts[:5]

    context_text = "\n\n".join(
        [f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    prompt = f"""
You are a retrieval-grounded QA assistant.

Answer the question using ONLY the provided context blocks.
Do not use outside knowledge.
Do not invent facts.
Do not make unsupported claims.

You should synthesize across multiple context blocks when needed.
If the context supports only part of the answer, provide only that supported part.
Refuse only if the context contains no meaningful evidence for answering the question.

Rules:
- Use only the provided context.
- Combine evidence across contexts when relevant.
- Cover all major parts of the question if supported.
- Each answer point must be grounded in one or more context blocks.
- Do not include unsupported conclusions.
- Do NOT include explanations, commentary, prefaces, or notes outside the required format.
- Do NOT add a "Note:" section.
- Do NOT say things like "the provided context says" or "based on the context".
- If there is not enough evidence to answer any meaningful part, reply exactly:
"{REFUSAL_TEXT}"

Output format for answerable questions:
Answer:
- <claim 1> [Context X]
- <claim 2> [Context Y]
- <claim 3> [Context X, Context Z]

Support:
[Context X], [Context Y], [Context Z]

Output format for unanswerable questions:
{REFUSAL_TEXT}

Context:
{context_text}

Question:
{query}
"""

    for attempt in range(2):
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2}
            )

            content = response.get("message", {}).get("content", "").strip()
            return content if content else REFUSAL_TEXT

        except Exception as e:
            if attempt == 0:
                print(f"Generator error on first attempt: {e}. Retrying once...")
                time.sleep(1)
            else:
                print(f"Generator failed after retry: {e}")
                return REFUSAL_TEXT