"""FastAPI layer over the RAG pipeline."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import IngestionError, RagPipeline
from .store import VectorStore

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class SourceResponse(BaseModel):
    citation: str
    document_title: str
    chunk_index: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    provider: str
    citations: list[str]
    sources: list[SourceResponse]
    latency_ms: int


class DocumentResponse(BaseModel):
    id: int
    title: str
    source: str
    chunk_count: int


class IngestResponse(BaseModel):
    document_id: int
    title: str
    chunk_count: int
    characters: int


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """One pipeline per process; the embedding model load is expensive."""
    return RagPipeline(store=VectorStore(os.environ.get('INDEX_PATH', 'index.db')))


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(
        title='RAG Document Assistant',
        description='Ask questions about your own documents, with citations.',
        version='1.0.0',
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # Tests inject a pipeline; production resolves the cached one per request.
    def resolve() -> RagPipeline:
        return pipeline or get_pipeline()

    @app.get('/health')
    def health(rag: RagPipeline = Depends(resolve)) -> dict:
        return {
            'status': 'ok',
            'embedding_provider': rag.embedder.name,
            'answer_provider': rag.answerer.name,
            'documents': len(rag.documents()),
            'chunks': rag.store.chunk_count(),
        }

    @app.get('/documents', response_model=list[DocumentResponse])
    def list_documents(rag: RagPipeline = Depends(resolve)):
        return [
            DocumentResponse(
                id=d.id, title=d.title, source=d.source, chunk_count=d.chunk_count
            )
            for d in rag.documents()
        ]

    @app.post('/documents', response_model=IngestResponse, status_code=201)
    async def upload(
        file: UploadFile = File(...), rag: RagPipeline = Depends(resolve)
    ):
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'File must be 10 MB or smaller.')

        try:
            result = rag.ingest_file(file.filename or 'document.txt', data)
        except IngestionError as exc:
            raise HTTPException(400, str(exc)) from exc

        return IngestResponse(
            document_id=result.document_id,
            title=result.title,
            chunk_count=result.chunk_count,
            characters=result.characters,
        )

    @app.delete('/documents/{document_id}', status_code=204)
    def delete_document(document_id: int, rag: RagPipeline = Depends(resolve)):
        if not rag.delete(document_id):
            raise HTTPException(404, f'No document with id {document_id}.')

    @app.post('/query', response_model=QueryResponse)
    def query(request: QueryRequest, rag: RagPipeline = Depends(resolve)):
        try:
            result = rag.query(request.question, limit=request.limit)
        except IngestionError as exc:
            raise HTTPException(400, str(exc)) from exc

        return QueryResponse(
            question=result.question,
            answer=result.answer.text,
            provider=result.answer.provider,
            citations=result.answer.citations,
            sources=[
                SourceResponse(
                    citation=hit.citation,
                    document_title=hit.document_title,
                    chunk_index=hit.chunk_index,
                    score=hit.score,
                    excerpt=hit.text[:280],
                )
                for hit in result.hits
            ],
            latency_ms=result.latency_ms,
        )

    return app


app = create_app()
