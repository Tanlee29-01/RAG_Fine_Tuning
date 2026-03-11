class VectorStore:
    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        self.ids = ids
        self.vectors = vectors
        self.metadatas = metadatas

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        return []
