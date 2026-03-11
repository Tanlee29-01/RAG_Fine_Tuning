from pathlib import Path


def load_pdfs(input_dir: str = "data/raw") -> list[Path]:
    return sorted(Path(input_dir).glob("*.pdf"))
