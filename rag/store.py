"""SQLite-backed vector store.

Embeddings live as float32 blobs alongside their chunk text, and similarity is
plain cosine over a normalised matrix. That is enough for corpora up to roughly
100k chunks and keeps the whole index in one portable file with no external
service to run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    start_char   INTEGER NOT NULL,
    end_char     INTEGER NOT NULL,
    embedding    BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


@dataclass(frozen=True)
class Document:
    id: int
    title: str
    source: str
    chunk_count: int


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk plus the score and provenance needed to cite it."""

    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    score: float

    @property
    def citation(self) -> str:
        return f'{self.document_title}#{self.chunk_index}'


class VectorStore:
    def __init__(self, path: str | Path = ':memory:') -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute('PRAGMA foreign_keys = ON')
        self._connection.executescript(SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def add_document(
        self,
        title: str,
        source: str,
        chunks: list,
        embeddings: np.ndarray,
    ) -> int:
        """Store a document and its chunks in one transaction."""
        if len(chunks) != len(embeddings):
            raise ValueError('Chunk count and embedding count must match.')

        with self._connection:
            cursor = self._connection.execute(
                'INSERT INTO documents (title, source) VALUES (?, ?)',
                (title, source),
            )
            document_id = int(cursor.lastrowid)
            self._connection.executemany(
                'INSERT INTO chunks '
                '(document_id, chunk_index, text, start_char, end_char, embedding) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                [
                    (
                        document_id,
                        chunk.index,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        vector.astype(np.float32).tobytes(),
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ],
            )
        return document_id

    def list_documents(self) -> list[Document]:
        rows = self._connection.execute(
            'SELECT d.id, d.title, d.source, COUNT(c.id) AS chunk_count '
            'FROM documents d LEFT JOIN chunks c ON c.document_id = d.id '
            'GROUP BY d.id ORDER BY d.id'
        ).fetchall()
        return [
            Document(
                id=row['id'],
                title=row['title'],
                source=row['source'],
                chunk_count=row['chunk_count'],
            )
            for row in rows
        ]

    def delete_document(self, document_id: int) -> bool:
        with self._connection:
            cursor = self._connection.execute(
                'DELETE FROM documents WHERE id = ?', (document_id,)
            )
        return cursor.rowcount > 0

    def chunk_count(self) -> int:
        return int(
            self._connection.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        )

    def search(
        self, query_vector: np.ndarray, limit: int = 5, min_score: float = 0.0
    ) -> list[SearchHit]:
        """Return the closest chunks by cosine similarity."""
        rows = self._connection.execute(
            'SELECT c.id, c.document_id, c.chunk_index, c.text, '
            '       c.start_char, c.end_char, c.embedding, d.title '
            'FROM chunks c JOIN documents d ON d.id = c.document_id'
        ).fetchall()

        # An empty index has no matrix to dot against; return early rather than
        # letting numpy raise on a zero-row array.
        if not rows:
            return []

        matrix = np.frombuffer(
            b''.join(row['embedding'] for row in rows), dtype=np.float32
        ).reshape(len(rows), -1)

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != matrix.shape[1]:
            raise ValueError(
                f'Query has {query.shape[0]} dimensions but the index has '
                f'{matrix.shape[1]}. Re-ingest after changing embedding model.'
            )

        # Rows are stored already normalised, so the dot product is the cosine.
        scores = matrix @ query

        ranked = np.argsort(scores)[::-1][:limit]
        return [
            SearchHit(
                chunk_id=rows[i]['id'],
                document_id=rows[i]['document_id'],
                document_title=rows[i]['title'],
                chunk_index=rows[i]['chunk_index'],
                text=rows[i]['text'],
                start_char=rows[i]['start_char'],
                end_char=rows[i]['end_char'],
                score=round(float(scores[i]), 4),
            )
            for i in ranked
            if float(scores[i]) >= min_score
        ]
