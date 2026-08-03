"""Split documents into overlapping chunks that keep sentences intact."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split on sentence-ending punctuation followed by whitespace.
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


class ChunkingError(ValueError):
    """Raised when a document cannot be chunked."""


@dataclass(frozen=True)
class Chunk:
    """One retrievable span of a document, with its position in the original."""

    index: int
    text: str
    start_char: int
    end_char: int


def split_sentences(text: str) -> list[str]:
    """Split into sentences, falling back to the whole text if none are found."""
    parts = [part.strip() for part in SENTENCE_END.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Break `text` into chunks of roughly `size` characters.

    Chunks are built from whole sentences so a retrieved span never starts or
    ends mid-thought. `overlap` characters of the previous chunk are carried
    into the next one, so an answer spanning a boundary is still retrievable.
    """
    if size <= 0:
        raise ChunkingError('Chunk size must be positive.')
    if overlap < 0:
        raise ChunkingError('Overlap cannot be negative.')
    if overlap >= size:
        raise ChunkingError('Overlap must be smaller than the chunk size.')

    cleaned = text.strip()
    if not cleaned:
        return []

    sentences = split_sentences(cleaned)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        if not buffer:
            return
        body = ' '.join(buffer)
        start = cleaned.find(body[:60], chunks[-1].start_char if chunks else 0)
        start = start if start >= 0 else 0
        chunks.append(
            Chunk(
                index=len(chunks),
                text=body,
                start_char=start,
                end_char=start + len(body),
            )
        )

    for sentence in sentences:
        # A single sentence longer than the chunk size is split on width; without
        # this the buffer could never drain and the loop would not terminate.
        if len(sentence) > size:
            flush()
            buffer, buffer_len = [], 0
            for offset in range(0, len(sentence), size):
                piece = sentence[offset : offset + size]
                start = cleaned.find(piece)
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        text=piece,
                        start_char=max(start, 0),
                        end_char=max(start, 0) + len(piece),
                    )
                )
            continue

        if buffer_len + len(sentence) > size and buffer:
            flush()
            tail = _tail(buffer, overlap)
            buffer = list(tail)
            buffer_len = sum(len(part) + 1 for part in tail)

        buffer.append(sentence)
        buffer_len += len(sentence) + 1

    flush()
    return chunks


def _tail(sentences: list[str], overlap: int) -> list[str]:
    """The trailing sentences that fit inside `overlap` characters."""
    kept: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > overlap:
            break
        kept.insert(0, sentence)
        total += len(sentence) + 1
    return kept
