def to_answer_schema(answer: str, citations: list[dict]) -> dict:
    return {"answer": answer, "citations": citations, "confidence": 0.5}
