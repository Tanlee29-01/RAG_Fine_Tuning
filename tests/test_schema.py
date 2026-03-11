from src.llm.response_validator import validate_response


def test_validate_response():
    assert validate_response({"answer": "ok", "citations": []})
