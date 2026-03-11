"""Ingest PDFs from data/raw/ — parse, clean, and prepare documents."""
from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.parser import parse_pdf
from src.preprocessing.cleaner import clean_text
from src.ingestion.metadata_builder import build_metadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ingest(input_dir: str = "data/raw") -> list[dict]:
    pdfs = load_pdfs(input_dir)
    if not pdfs:
        logger.warning("No PDF files found in %s", input_dir)
        return []

    docs = []
    for pdf in pdfs:
        logger.info("Parsing %s", pdf.name)
        doc = parse_pdf(str(pdf))
        doc["text"] = clean_text(doc.get("text", ""))
        doc.update(build_metadata(str(pdf)))
        docs.append(doc)

    logger.info("Ingested %d documents", len(docs))
    return docs


if __name__ == "__main__":
    results = ingest()
    for d in results:
        print(d.get("doc_id"), d.get("title"))
