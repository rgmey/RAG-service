# app/rag/ingestion.py
from pypdf import PdfReader


def extract_text(file_path: str) -> str:
    """Extracts all text from a PDF, page by page, joined with newlines."""
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()
