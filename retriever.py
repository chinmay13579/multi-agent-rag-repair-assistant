"""
BM25-based text retriever — LangChain edition.

Uses langchain_community.retrievers.BM25Retriever (backed by rank-bm25)
and wraps it to expose the same interface as the original retriever so the
rest of the codebase needs zero changes.
"""

from langchain_community.retrievers import BM25Retriever as _LCBm25

import config


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        """
        Build a BM25 index from a list of chunk dicts.
        Each chunk: {"chunk_id": int, "page": int, "text": str}
        """
        self.chunks = chunks

        texts     = [c["text"] for c in chunks]
        metadatas = [{"chunk_id": c["chunk_id"], "page": c["page"]} for c in chunks]
        if not texts:
             raise ValueError(
        "No text extracted from the PDF. "
        "Ensure Tesseract OCR is installed and 'tesseract' is in your system PATH."
    )

        self._lc_retriever = _LCBm25.from_texts(texts, metadatas=metadatas)
        self._lc_retriever.k = config.TOP_K_CHUNKS

    def retrieve(self, query: str, top_k: int = config.TOP_K_CHUNKS) -> list[dict]:
        """
        Return the top_k most relevant chunks for the query.
        Each returned dict has: chunk_id, page, text, score, rank.

        Note: LangChain's BM25Retriever does not expose raw BM25 scores,
        so we use a normalised rank-based proxy (1/rank).
        """
        self._lc_retriever.k = top_k
        docs = self._lc_retriever.invoke(query)

        results = []
        for rank, doc in enumerate(docs, start=1):
            meta     = doc.metadata
            chunk_id = meta.get("chunk_id", -1)
            results.append({
                "chunk_id": chunk_id,
                "page":     meta.get("page", 0),
                "text":     doc.page_content,
                "score":    round(1.0 / rank, 4),   # proxy: rank 1 → 1.0, rank 2 → 0.5 …
                "rank":     rank,
            })
        return results

    def get_pages_for_query(self, query: str, top_k: int = config.TOP_K_CHUNKS) -> list[int]:
        """Return unique page numbers of the top retrieved chunks."""
        results = self.retrieve(query, top_k=top_k)
        seen: set = set()
        pages: list[int] = []
        for r in results:
            if r["page"] not in seen:
                seen.add(r["page"])
                pages.append(r["page"])
        return pages
