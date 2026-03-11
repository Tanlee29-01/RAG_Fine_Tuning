def build_prompt(question: str, context: str) -> str:
    return f"Context:
{context}

Question: {question}
Answer with citations."
