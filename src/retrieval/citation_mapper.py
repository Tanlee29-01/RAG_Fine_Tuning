def map_citations(chunks: list[dict]) -> list[dict]:
    return [{"doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id")} for c in chunks]
