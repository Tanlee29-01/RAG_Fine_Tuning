from src.chunking.chunker import chunk_text


def test_chunk_text_returns_chunks():
    chunks = chunk_text("a" * 2000, chunk_size=200, overlap=50)
    assert len(chunks) > 1
