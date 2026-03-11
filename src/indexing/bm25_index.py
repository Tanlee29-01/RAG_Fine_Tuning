class BM25Index:
    def add(self, texts: list[str]) -> None:
        self.texts = texts

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return []
