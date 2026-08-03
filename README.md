# RAG Document Assistant

Ask questions about your own documents and get answers **with citations back to
the exact passage**. Upload PDFs or Markdown, and every sentence in the answer
points at the chunk it came from.

**FastAPI + PyTorch (sentence-transformers) + SQLite.** No vector database to
run, no API key required.

---

## Runs offline by default

The pipeline is built around two provider interfaces, each with a local default
and a hosted upgrade:

| Layer | Default (no key) | With `ANTHROPIC_API_KEY` |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local, 384-d | same |
| Answering | **Extractive** — returns retrieved sentences verbatim | **Claude** — fluent prose, grounded in the passages |

The extractive answerer cannot hallucinate: every sentence it returns appears
word for word in an indexed document. That makes the demo trustworthy with zero
setup, and switching to Claude is one environment variable — the retrieval layer
does not change.

```bash
export ANTHROPIC_API_KEY=sk-...   # optional; extractive is the default
```

---

## Measured behaviour

From `demo.py` on the bundled 3-document corpus (4 chunks), local model on CPU:

| Question | Retrieved | Top score | Latency |
|---|---|---|---|
| "What is the standard deposit for a lease?" | `tenancy-handbook#0` | 0.620 | 165 ms |
| "How quickly are emergency repairs handled?" | `service-charter#0` | 0.720 | 73 ms |
| "Can I keep a dog in a third floor flat?" | `tenancy-handbook#1` | 0.573 | 83 ms |
| "What is the refund policy for annual plans?" | `billing-policy#0` | 0.416 | 79 ms |

The third question is the interesting one: the corpus never says "third floor."
Retrieval still lands on the pets clause ("ground floor units only"), which is
the passage a person needs in order to answer it.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python demo.py                    # end-to-end over sample_docs/
uvicorn rag.api:app --reload      # API on http://localhost:8000
pytest tests -q                   # 48 tests
```

Interactive API docs at `http://localhost:8000/docs`.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Index state and which providers are active |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/documents` | Upload a PDF / Markdown / text file |
| `DELETE` | `/documents/{id}` | Remove a document and its chunks |
| `POST` | `/query` | Ask a question; returns answer + ranked sources |

```bash
curl -F "file=@sample_docs/tenancy-handbook.md" http://localhost:8000/documents

curl -X POST -H "Content-Type: application/json" \
     -d '{"question":"how much is the deposit?","limit":3}' \
     http://localhost:8000/query
```

Every `/query` response carries a `sources` array: citation label, document
title, chunk index, cosine score, and a 280-character excerpt — enough for a UI
to show *why* the model said what it said.

---

## Design decisions worth explaining

**Sentence-aware chunking with overlap.** Chunks are assembled from whole
sentences so a retrieved span never starts mid-thought, with 150 characters of
overlap so an answer straddling a boundary is still findable. A sentence longer
than the chunk size is split on width — without that case the buffer can never
drain and chunking does not terminate.

**Cosine similarity in ~15 lines instead of a vector database.** Embeddings are
stored as float32 blobs in SQLite and rows are pre-normalised, so the dot
product *is* the cosine. This handles corpora up to roughly 100k chunks, keeps
the index in one portable file, and means the retrieval maths is auditable
rather than delegated.

**Provider boundaries.** `rag/` knows nothing about HTTP; `rag/api.py` knows
nothing about embeddings or model internals. Swapping the embedding model,
answer model, or transport touches one file each.

**Citations carry character offsets.** Each hit records `start_char`/`end_char`
in the source document, so a UI can highlight the exact span rather than just
naming the file.

---

## Things the code handles that a demo usually doesn't

- Mixed encodings on upload (UTF-8 → Latin-1 fallback)
- Empty index queried before ingestion (returns an honest "not found", not a crash)
- Dimension mismatch after switching embedding model (clear error naming the fix)
- Markdown headings stripped from answers but kept in the retrieval signal
- Duplicate sentences from overlapping chunks collapsed in the answer
- Claude refusals (HTTP 200 with empty content) checked before reading the response

---

## Layout

```
rag-document-assistant/
├── rag/
│   ├── chunking.py     # sentence-aware splitting with overlap
│   ├── embeddings.py   # provider protocol + local / hashing implementations
│   ├── store.py        # SQLite vector store, cosine search
│   ├── answering.py    # extractive and Claude answerers
│   ├── pipeline.py     # ingest / search / query — no HTTP awareness
│   └── api.py          # FastAPI layer
├── tests/test_rag.py   # 48 tests
├── sample_docs/
└── demo.py
```

## License

MIT
