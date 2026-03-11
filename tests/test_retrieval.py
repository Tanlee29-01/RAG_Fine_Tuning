from src.retrieval.context_builder import build_context


def test_build_context_non_empty():
    context = build_context([{"text": "hello"}, {"text": "world"}])
    assert "hello" in context
