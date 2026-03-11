def split_sections(text: str) -> list[str]:
    return [s.strip() for s in text.split("\n\n") if s.strip()]
