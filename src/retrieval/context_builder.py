def build_context(chunks: list[dict], max_chars: int = 12000) -> str:
    parts, total = [], 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)
