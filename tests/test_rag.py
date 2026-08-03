import io
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag.answering import ExtractiveAnswerer, build_context, clean_sentence
from rag.api import create_app
from rag.chunking import ChunkingError, chunk_text, split_sentences
from rag.embeddings import HashingEmbeddingProvider, normalise, resolve_provider
from rag.embeddings import EmbeddingError
from rag.pipeline import IngestionError, RagPipeline, decode_text
from rag.store import VectorStore

HANDBOOK = (
    'Tenants must give thirty days notice before ending a lease. '
    'The standard deposit is one month of rent. '
    'Deposits are returned within fourteen days of a satisfactory inspection. '
    'Pets are permitted in ground floor units only.'
)

CHARTER = (
    'Support requests are answered within one business day. '
    'Emergency repairs are attended to within four hours. '
    'Routine maintenance is scheduled within one week of the request.'
)


@pytest.fixture
def embedder():
    # The hashing provider keeps tests fast, offline, and deterministic.
    return HashingEmbeddingProvider(dimensions=256)


@pytest.fixture
def pipeline(embedder):
    store = VectorStore()
    rag = RagPipeline(
        store=store, embedder=embedder, answerer=ExtractiveAnswerer(embedder)
    )
    yield rag
    store.close()


@pytest.fixture
def loaded(pipeline):
    pipeline.ingest('tenancy-handbook', 'handbook.md', HANDBOOK)
    pipeline.ingest('service-charter', 'charter.md', CHARTER)
    return pipeline


@pytest.fixture
def client(loaded):
    return TestClient(create_app(loaded))


class TestChunking:
    def test_short_text_is_a_single_chunk(self):
        chunks = chunk_text(HANDBOOK, size=900)
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_long_text_splits_into_several_chunks(self):
        chunks = chunk_text(HANDBOOK * 12, size=400, overlap=80)
        assert len(chunks) > 1
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_chunks_respect_the_size_budget(self):
        # Overlap can carry a little past the target, but not unboundedly.
        for chunk in chunk_text(HANDBOOK * 12, size=400, overlap=80):
            assert len(chunk.text) <= 400 + 80

    def test_sentence_longer_than_chunk_size_terminates(self):
        # A run of text with no sentence break used to loop forever.
        chunks = chunk_text('x' * 2000, size=300, overlap=50)
        assert chunks
        assert all(len(chunk.text) <= 300 for chunk in chunks)

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text('   \n  ') == []

    def test_rejects_overlap_larger_than_size(self):
        with pytest.raises(ChunkingError):
            chunk_text(HANDBOOK, size=100, overlap=100)

    def test_rejects_non_positive_size(self):
        with pytest.raises(ChunkingError):
            chunk_text(HANDBOOK, size=0)

    def test_split_sentences_falls_back_to_whole_text(self):
        assert split_sentences('no punctuation here') == ['no punctuation here']


class TestEmbeddings:
    def test_vectors_are_unit_length(self, embedder):
        vectors = embedder.embed(['hello world', 'another document'])
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_embedding_is_deterministic(self, embedder):
        assert np.array_equal(embedder.embed(['repeat']), embedder.embed(['repeat']))

    def test_similar_text_scores_higher_than_unrelated(self, embedder):
        base, similar, unrelated = embedder.embed(
            [
                'the deposit is one month of rent',
                'deposit equal to one month rent',
                'emergency repairs within four hours',
            ]
        )
        assert float(base @ similar) > float(base @ unrelated)

    def test_normalise_leaves_zero_vectors_finite(self):
        result = normalise(np.zeros((1, 8), dtype=np.float32))
        assert np.all(np.isfinite(result))

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(EmbeddingError):
            resolve_provider('not-a-provider')


class TestVectorStore:
    def test_search_on_empty_index_returns_nothing(self, embedder):
        store = VectorStore()
        assert store.search(embedder.embed(['anything'])[0]) == []

    def test_documents_report_their_chunk_counts(self, loaded):
        documents = loaded.documents()
        assert {d.title for d in documents} == {'tenancy-handbook', 'service-charter'}
        assert all(d.chunk_count > 0 for d in documents)

    def test_results_are_ordered_by_score(self, loaded):
        hits = loaded.search('how long until my deposit comes back?', limit=4)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_dimension_mismatch_is_reported_clearly(self, loaded):
        with pytest.raises(ValueError, match='dimensions'):
            loaded.store.search(np.zeros(7, dtype=np.float32))

    def test_deleting_a_document_removes_its_chunks(self, loaded):
        before = loaded.store.chunk_count()
        target = loaded.documents()[0]

        assert loaded.delete(target.id) is True
        assert loaded.store.chunk_count() < before
        assert target.id not in {d.id for d in loaded.documents()}

    def test_deleting_a_missing_document_reports_false(self, loaded):
        assert loaded.delete(9999) is False

    def test_mismatched_chunk_and_embedding_counts_are_rejected(self, embedder):
        store = VectorStore()
        chunks = chunk_text(HANDBOOK)
        with pytest.raises(ValueError):
            store.add_document('t', 's', chunks, embedder.embed(['a', 'b', 'c']))


class TestRetrieval:
    def test_retrieves_the_relevant_document(self, loaded):
        hits = loaded.search('when are emergency repairs handled?', limit=3)
        assert hits[0].document_title == 'service-charter'

    def test_citation_identifies_document_and_chunk(self, loaded):
        hit = loaded.search('deposit', limit=1)[0]
        assert hit.citation == f'{hit.document_title}#{hit.chunk_index}'

    def test_offsets_point_back_into_the_source_text(self, loaded):
        hit = loaded.search('deposit', limit=1)[0]
        assert hit.end_char > hit.start_char

    def test_limit_is_honoured(self, loaded):
        assert len(loaded.search('notice', limit=2)) <= 2

    def test_empty_question_is_rejected(self, loaded):
        with pytest.raises(IngestionError):
            loaded.search('   ')


class TestAnswering:
    def test_answer_cites_its_source(self, loaded):
        result = loaded.query('how much is the deposit?')
        assert result.answer.citations
        assert '[' in result.answer.text

    def test_every_answer_sentence_comes_from_the_corpus(self, loaded):
        result = loaded.query('how much is the deposit?')
        corpus = f'{HANDBOOK} {CHARTER}'
        for sentence in result.answer.text.split(' ['):
            fragment = sentence.split(']')[-1].strip()
            if len(fragment) > 20:
                assert fragment in corpus

    def test_empty_index_answers_honestly(self, pipeline):
        result = pipeline.query('anything at all?')
        assert result.hits == []
        assert "couldn't find" in result.answer.text

    def test_latency_is_recorded(self, loaded):
        assert loaded.query('deposit').latency_ms >= 0

    def test_markdown_headings_are_stripped_from_answers(self, pipeline):
        pipeline.ingest(
            'policy',
            'policy.md',
            '# Refunds\n\n## Annual plans\n\n'
            'Refunds on annual plans are pro rated for unused months. '
            'Processing takes ten working days from confirmation.',
        )
        text = pipeline.query('how are annual refunds calculated?').answer.text
        # Strip the [doc#n] citation labels; '#' is legitimate inside those.
        body = re.sub(r'\[[^\]]*\]', '', text)

        assert '#' not in body
        assert 'Refunds' in body
        assert 'pro rated' in body

    def test_overlapping_chunks_do_not_repeat_a_sentence(self, pipeline):
        # Small chunks with large overlap guarantee duplicated sentences.
        pipeline.chunk_size, pipeline.chunk_overlap = 120, 90
        pipeline.ingest('doc', 'doc.md', HANDBOOK)

        text = pipeline.query('what is the deposit?').answer.text
        sentences = [s.split(']')[-1].strip() for s in text.split(' [')]
        meaningful = [s for s in sentences if len(s) > 20]
        assert len(meaningful) == len(set(meaningful))

    def test_clean_sentence_removes_heading_lines(self):
        assert clean_sentence('## Repairs\nFixed within four hours.') == (
            'Fixed within four hours.'
        )

    def test_context_is_labelled_for_citation(self, loaded):
        hits = loaded.search('deposit', limit=2)
        context = build_context(hits)
        for hit in hits:
            assert f'[{hit.citation}]' in context


class TestIngestion:
    def test_ingest_reports_chunk_count(self, pipeline):
        result = pipeline.ingest('doc', 'doc.md', HANDBOOK * 8)
        assert result.chunk_count > 1
        assert result.characters == len(HANDBOOK * 8)

    def test_blank_document_is_rejected(self, pipeline):
        with pytest.raises(IngestionError):
            pipeline.ingest('doc', 'doc.md', '    ')

    def test_untitled_document_is_rejected(self, pipeline):
        with pytest.raises(IngestionError):
            pipeline.ingest('  ', 'doc.md', HANDBOOK)

    def test_unsupported_file_type_is_rejected(self, pipeline):
        with pytest.raises(IngestionError, match='Unsupported'):
            pipeline.ingest_file('archive.zip', b'PK\x03\x04')

    def test_latin1_text_is_decoded(self):
        assert 'café' in decode_text('café résumé'.encode('latin-1'))

    def test_markdown_upload_is_ingested(self, pipeline):
        result = pipeline.ingest_file('notes.md', HANDBOOK.encode())
        assert result.title == 'notes'
        assert result.chunk_count >= 1


class TestApi:
    def test_health_reports_index_state(self, client):
        body = client.get('/health').json()
        assert body['status'] == 'ok'
        assert body['documents'] == 2
        assert body['chunks'] > 0

    def test_documents_are_listed(self, client):
        response = client.get('/documents')
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_query_returns_answer_and_sources(self, client):
        response = client.post('/query', json={'question': 'deposit amount?'})

        assert response.status_code == 200
        body = response.json()
        assert body['answer']
        assert body['sources']
        assert body['sources'][0]['citation']

    def test_query_rejects_an_empty_question(self, client):
        assert client.post('/query', json={'question': ''}).status_code == 422

    def test_query_rejects_an_out_of_range_limit(self, client):
        response = client.post('/query', json={'question': 'x', 'limit': 500})
        assert response.status_code == 422

    def test_upload_then_query_the_new_document(self, client):
        upload = client.post(
            '/documents',
            files={'file': ('policy.md', io.BytesIO(b'Refunds take five days.'), 'text/markdown')},
        )
        assert upload.status_code == 201

        body = client.post('/query', json={'question': 'how long do refunds take?'}).json()
        assert 'policy' in ' '.join(body['citations'])

    def test_upload_rejects_unsupported_type(self, client):
        response = client.post(
            '/documents',
            files={'file': ('a.zip', io.BytesIO(b'PK'), 'application/zip')},
        )
        assert response.status_code == 400

    def test_delete_removes_the_document(self, client):
        document_id = client.get('/documents').json()[0]['id']

        assert client.delete(f'/documents/{document_id}').status_code == 204
        assert len(client.get('/documents').json()) == 1

    def test_deleting_a_missing_document_is_404(self, client):
        assert client.delete('/documents/4321').status_code == 404
