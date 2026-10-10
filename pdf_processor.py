"""
PDF Processor — extracts text chunks and renders pages as images.

Text extraction  : pdfplumber  (text + tables)
Image rendering  : PyMuPDF / fitz  (page → PNG for vision model)
"""

import json
import os
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from typing import Optional
import pytesseract

import config
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\tesseract.exe'

def _words(text: str) -> list[str]:
    return text.split()


def extract_chunks(pdf_path: str) -> list[dict]:
    """
    Extract text from every page, include any tables as plain text,
    then split into overlapping word-count chunks.

    Returns a list of dicts:
        {"chunk_id": int, "page": int, "text": str}
    """
    cache = Path(config.CHUNKS_CACHE)
    if cache.exists():
        print(f"[pdf_processor] Loading cached chunks from {cache}")
        with cache.open() as f:
            return json.load(f)

    print(f"[pdf_processor] Extracting text from {pdf_path} …")
    all_page_texts: list[tuple[int, str]] = []  # (page_num, text)

    # First try pdfplumber for native text; fall back to OCR for image-based pages
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(2.0, 2.0)  # render at 2× for better OCR accuracy

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            parts: list[str] = []

            # 1. Try native text extraction
            raw = page.extract_text()
            if raw and raw.strip():
                parts.append(raw.strip())

            # 2. Try native table extraction
            tables = page.extract_tables()
            for table in tables:
                rows = []
                for row in table:
                    if row:
                        cells = [cell or "" for cell in row]
                        if any(c.strip() for c in cells):
                            rows.append("\t".join(cells))
                if rows:
                    parts.append("[TABLE]\n" + "\n".join(rows))

            # 3. If no native text found, fall back to OCR (if tesseract is installed)
            if not parts :
                fitz_page = doc[page_num - 1]
                pix = fitz_page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang="eng").strip()
                if ocr_text:
                    parts.append(ocr_text)

            if parts:
                all_page_texts.append((page_num, "\n\n".join(parts)))

    doc.close()

    # Split each page text into overlapping chunks
    chunks: list[dict] = []
    size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    for page_num, text in all_page_texts:
        words = _words(text)
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + size
            chunk_text = " ".join(words[start:end])
            chunks.append(
                {"chunk_id": len(chunks), "page": page_num, "text": chunk_text}
            )
            if end >= len(words):
                break
            start += size - overlap

    # Cache to disk
    Path(config.CACHE_DIR).mkdir(exist_ok=True)
    with cache.open("w") as f:
        json.dump(chunks, f, indent=2)

    print(f"[pdf_processor] Extracted {len(chunks)} chunks from {len(all_page_texts)} pages.")
    return chunks


def render_pages(pdf_path: str) -> list[dict]:
    """
    Render every PDF page to a PNG file (for the vision model).

    Returns a list of dicts:
        {"page": int, "path": str}
    """
    index_path = Path(config.IMAGE_INDEX_CACHE)
    if index_path.exists():
        print(f"[pdf_processor] Loading cached image index from {index_path}")
        with index_path.open() as f:
            return json.load(f)

    print(f"[pdf_processor] Rendering pages from {pdf_path} …")
    Path(config.IMAGES_DIR).mkdir(exist_ok=True)
    Path(config.CACHE_DIR).mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(config.IMAGE_DPI_SCALE, config.IMAGE_DPI_SCALE)
    image_index: list[dict] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(config.IMAGES_DIR, f"page_{page_num + 1}.png")
        pix.save(img_path)
        image_index.append({"page": page_num + 1, "path": img_path})

    doc.close()

    with index_path.open("w") as f:
        json.dump(image_index, f, indent=2)

    print(f"[pdf_processor] Rendered {len(image_index)} page images.")
    return image_index


def get_page_image_path(image_index: list[dict], page_num: int) -> Optional[str]:
    """Return the file path of the rendered image for a given page number."""
    for entry in image_index:
        if entry["page"] == page_num:
            return entry["path"]
    return None
