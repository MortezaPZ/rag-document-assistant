"""Answer generation over retrieved chunks.

The extractive answerer is the default: it needs no API key and cannot invent
facts, because every sentence it returns is copied from a retrieved chunk. The
Claude answerer produces fluent prose and is selected by setting an API key.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .chunking import split_sentences
from .embeddings import EmbeddingProvider, normalise
from .store import SearchHit

HEADING = re.compile(r'^#{1,6}\s')

SYSTEM_PROMPT = (
    'You answer questions strictly from the supplied context passages. '
    'Cite the passage you used with its bracketed label, for example [handbook#2]. '
    'If the passages do not contain the answer, say so plainly instead of guessing.'
)


class AnsweringError(RuntimeError):
    """Raised when an answer cannot be produced."""


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]
    provider: str


@runtime_checkable
class Answerer(Protocol):
    name: str

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        ...


def clean_sentence(sentence: str) -> str:
    """Drop markdown chrome so a heading never reads as part of the answer.

    Retrieval keeps headings because they carry topic signal, but quoting
    "## Repairs" back at the user is noise.
    """
    lines = [
        line.strip()
        for line in sentence.splitlines()
        if line.strip() and not HEADING.match(line.strip())
    ]
    return ' '.join(lines).strip()


def build_context(hits: list[SearchHit]) -> str:
    """Render retrieved chunks as labelled passages the model can cite."""
    return '\n\n'.join(f'[{hit.citation}]\n{hit.text}' for hit in hits)


class ExtractiveAnswerer:
    """Return the retrieved sentences that best match the question.

    No generation, so no hallucination: every returned sentence appears verbatim
    in an indexed document.
    """

    name = 'extractive'

    def __init__(self, embedder: EmbeddingProvider, max_sentences: int = 3) -> None:
        self.embedder = embedder
        self.max_sentences = max_sentences

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(
                text="I couldn't find anything relevant in the indexed documents.",
                citations=[],
                provider=self.name,
            )

        candidates: list[tuple[str, str]] = []
        seen_sentences: set[str] = set()
        for hit in hits:
            for sentence in split_sentences(hit.text):
                cleaned = clean_sentence(sentence)
                # Chunk overlap means the same sentence can arrive from two
                # hits; keep the first and drop the repeat.
                key = cleaned.lower()
                if len(cleaned) > 15 and key not in seen_sentences:
                    seen_sentences.add(key)
                    candidates.append((cleaned, hit.citation))

        if not candidates:
            return Answer(
                text=hits[0].text, citations=[hits[0].citation], provider=self.name
            )

        question_vector = self.embedder.embed([question])[0]
        sentence_matrix = normalise(
            self.embedder.embed([sentence for sentence, _ in candidates])
        )
        scores = sentence_matrix @ question_vector

        best = sorted(
            range(len(candidates)), key=lambda i: float(scores[i]), reverse=True
        )[: self.max_sentences]
        # Restore document order so the answer reads as continuous prose.
        best.sort()

        sentences = [f'{candidates[i][0]} [{candidates[i][1]}]' for i in best]
        seen: list[str] = []
        for i in best:
            citation = candidates[i][1]
            if citation not in seen:
                seen.append(citation)

        return Answer(text=' '.join(sentences), citations=seen, provider=self.name)


class ClaudeAnswerer:
    """Generate an answer with Claude, grounded in the retrieved passages."""

    name = 'claude'

    def __init__(self, model: str = 'claude-opus-5') -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise AnsweringError(
                'The anthropic package is not installed. '
                'Install it, or unset ANTHROPIC_API_KEY to use the extractive answerer.'
            ) from exc

        self.model = model
        self._client = anthropic.Anthropic()

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        if not hits:
            return Answer(
                text="I couldn't find anything relevant in the indexed documents.",
                citations=[],
                provider=self.name,
            )

        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    'role': 'user',
                    'content': (
                        f'Context passages:\n\n{build_context(hits)}\n\n'
                        f'Question: {question}'
                    ),
                }
            ],
        )

        # A refusal returns a 200 with empty content, so check before indexing.
        if message.stop_reason == 'refusal':
            raise AnsweringError('The model declined to answer this question.')

        text = ''.join(
            block.text for block in message.content if block.type == 'text'
        )
        cited = [hit.citation for hit in hits if hit.citation in text]
        return Answer(
            text=text.strip(),
            citations=cited or [hit.citation for hit in hits],
            provider=self.name,
        )


def resolve_answerer(embedder: EmbeddingProvider, name: str | None = None) -> Answerer:
    """Use Claude when an API key is configured, otherwise stay extractive."""
    choice = (name or os.environ.get('ANSWER_PROVIDER') or '').lower()
    if not choice:
        choice = 'claude' if os.environ.get('ANTHROPIC_API_KEY') else 'extractive'

    if choice == 'extractive':
        return ExtractiveAnswerer(embedder)
    if choice == 'claude':
        return ClaudeAnswerer(os.environ.get('ANSWER_MODEL', 'claude-opus-5'))
    raise AnsweringError(f'Unknown answer provider "{choice}".')
