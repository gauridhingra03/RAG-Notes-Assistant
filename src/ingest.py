# src/ingest.py
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import MAX_PAGES


def load_pdf_pages(pdf_path: str, max_pages: int = MAX_PAGES) -> list[dict]:
    """PDF ko page-wise text mein load karta hai — page number metadata mein rehta hai,
    text ke andar embed nahi hota, isliye reliably track ho sakta hai."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    if total_pages > max_pages:
        print(f"Warning: PDF has {total_pages} pages, processing only first {max_pages} pages.")
    pages = []
    for page_num, page in enumerate(doc):
        if page_num >= max_pages:
            break
        pages.append({"page": page_num + 1, "text": page.get_text()})
    doc.close()
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 600, chunk_overlap: int = 75) -> list[dict]:
    """Har page ko alag chunk karta hai, taaki har chunk ka page number pakka pata rahe."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    cid = 0
    for p in pages:
        page_chunks = splitter.split_text(p["text"])
        for c in page_chunks:
            if c.strip():
                chunks.append({"id": cid, "page": p["page"], "text": c})
                cid += 1
    return chunks