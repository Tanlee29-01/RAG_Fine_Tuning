def validate_response(payload: dict) -> bool:
    return "answer" in payload and "citations" in payload
