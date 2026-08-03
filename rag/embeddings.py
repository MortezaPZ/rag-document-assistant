"""Embedding providers.

The local provider is the default so the project runs with no API key and no
network access. `resolve_provider` swaps in a hosted provider when one is
configured, and the rest of the codebase only sees the `EmbeddingProvider`
protocol.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_LOCAL_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
TOKEN_PATTERN = re.compile(r'[a-z0-9]+')


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that turns text into unit-length vectors."""

    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dimensions) array of unit-length rows."""


def normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction; leaving the norm at 1 keeps it at zero
    # instead of producing NaN.
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbeddingProvider:
    """Deterministic bag-of-words embeddings with no model download.

    Used by the test suite and as the offline fallback. Quality is well below a
    trained model, but it is dependency-free, instant, and reproducible.
    """

    name = 'hashing'

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise EmbeddingError('Dimensions must be positive.')
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_PATTERN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                column = int.from_bytes(digest, 'big') % self.dimensions
                matrix[row, column] += 1.0
        return normalise(matrix)


class SentenceTransformerProvider:
    """Local transformer embeddings via sentence-transformers (PyTorch)."""

    name = 'sentence-transformers'

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EmbeddingError(
                'sentence-transformers is not installed. '
                'Install it, or set EMBEDDING_PROVIDER=hashing.'
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.astype(np.float32)


def resolve_provider(name: str | None = None) -> EmbeddingProvider:
    """Pick a provider from the environment, defaulting to the local model."""
    choice = (name or os.environ.get('EMBEDDING_PROVIDER') or 'sentence-transformers').lower()

    if choice == 'hashing':
        return HashingEmbeddingProvider()
    if choice == 'sentence-transformers':
        return SentenceTransformerProvider(
            os.environ.get('EMBEDDING_MODEL', DEFAULT_LOCAL_MODEL)
        )
    raise EmbeddingError(f'Unknown embedding provider "{choice}".')
