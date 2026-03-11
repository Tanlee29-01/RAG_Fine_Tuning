from pathlib import Path


def build_metadata(source_path: str) -> dict:
    p = Path(source_path)
    return {"doc_id": p.stem, "title": p.stem.replace("_", " ").title()}
