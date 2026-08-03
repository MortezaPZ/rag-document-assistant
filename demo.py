"""End-to-end demo: ingest the sample corpus and run a few questions."""

from pathlib import Path

from rag.pipeline import RagPipeline
from rag.store import VectorStore

QUESTIONS = [
    'What is the standard deposit for a lease?',
    'How quickly are emergency repairs handled?',
    'Can I keep a dog in a third floor flat?',
    'What is the refund policy for annual plans?',
]


def main() -> None:
    pipeline = RagPipeline(store=VectorStore())
    print(f'embeddings: {pipeline.embedder.name} ({pipeline.embedder.dimensions}d)')
    print(f'answers:    {pipeline.answerer.name}\n')

    for path in sorted(Path('sample_docs').glob('*.md')):
        result = pipeline.ingest(path.stem, path.name, path.read_text(encoding='utf-8'))
        print(f'ingested {result.title}: {result.chunk_count} chunks')

    print(f'\nindex holds {pipeline.store.chunk_count()} chunks\n')

    for question in QUESTIONS:
        result = pipeline.query(question, limit=3)
        print(f'Q: {question}')
        print(f'A: {result.answer.text}')
        print(f'   sources: {", ".join(result.answer.citations) or "none"}')
        print(f'   top score: {result.hits[0].score if result.hits else 0.0}')
        print(f'   latency: {result.latency_ms}ms\n')


if __name__ == '__main__':
    main()
