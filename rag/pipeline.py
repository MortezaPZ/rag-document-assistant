"""The RAG pipeline: ingest documents, then answer questions over them.

This module is the only thing the API layer talks to, and it knows nothing
about HTTP. That keeps retrieval testable on its own and lets the transport
change without touching the model logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .answering import Answer, Answerer, resolve_answerer
from .chunking import ChunkingError, chunk_text
from .embeddings import EmbeddingProvider, resolve_provider
from .store import Document, SearchHit, VectorStore


class IngestionError(ValueError):
    """Raised when a document cannot be ingested."""


@dataclass(frozen=True)
class IngestionResult:
    document_id: int
    title: str
    chunk_count: int
    characters: int


@dataclass(frozen=True)
class QueryResult:
    question: str
    answer: Answer
    hits: list[SearchHit]
    latency_ms: int


def read_pdf(data: bytes) -> str:
    """Extract text from a PDF, page by page."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise IngestionError('pypdf is not installed; cannot read PDFs.') from exc

    import io

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise IngestionError(f'Could not read the PDF: {exc}') from exc

    return '\n\n'.join(page.extract_text() or '' for page in reader.pages)


def decode_text(data: bytes) -> str:
    """Decode bytes, tolerating the encodings exported documents carry."""
    for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestionError('Could not decode the file as text.')


class RagPipeline:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        answerer: Answerer | None = None,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.store = store or VectorStore()
        self.embedder = embedder or resolve_provider()
        self.answerer = answerer or resolve_answerer(self.embedder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, title: str, source: str, text: str) -> IngestionResult:
        """Chunk, embed, and store one document."""
        if not title.strip():
            raise IngestionError('A document title is required.')

        try:
            chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        except ChunkingError as exc:
            raise IngestionError(str(exc)) from exc

        if not chunks:
            raise IngestionError('The document has no readable text.')

        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        document_id = self.store.add_document(title, source, chunks, embeddings)

        return IngestionResult(
            document_id=document_id,
            title=title,
            chunk_count=len(chunks),
            characters=len(text),
        )

    def ingest_file(self, filename: str, data: bytes) -> IngestionResult:
        """Ingest an uploaded PDF or text file."""
        suffix = Path(filename).suffix.lower()
        if suffix == '.pdf':
            text = read_pdf(data)
        elif suffix in {'.txt', '.md', '.markdown', ''}:
            text = decode_text(data)
        else:
            raise IngestionError(f'Unsupported file type "{suffix}".')
        return self.ingest(Path(filename).stem, filename, text)

    def search(self, question: str, limit: int = 5) -> list[SearchHit]:
        if not question.strip():
            raise IngestionError('The question cannot be empty.')
        vector = self.embedder.embed([question])[0]
        return self.store.search(vector, limit=limit)

    def query(self, question: str, limit: int = 5) -> QueryResult:
        """Retrieve the best chunks and answer from them."""
        started = time.perf_counter()
        hits = self.search(question, limit=limit)
        answer = self.answerer.answer(question, hits)
        return QueryResult(
            question=question,
            answer=answer,
            hits=hits,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def documents(self) -> list[Document]:
        return self.store.list_documents()

    def delete(self, document_id: int) -> bool:
        return self.store.delete_document(document_id)
