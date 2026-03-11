from src.ingestion.pdf_loader import load_pdfs


if __name__ == "__main__":
    print([str(p) for p in load_pdfs()])
