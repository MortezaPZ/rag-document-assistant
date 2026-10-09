# RAG Document Assistant

Ask a question about your own documents and get an answer that points back to the passage it came from.

## Overview

Upload a PDF, Markdown file, or plain text. The assistant retrieves the relevant sentences and either returns those sentences unchanged or, if you configure a key, asks a hosted model to write from the same passages.

The default path needs no API key and cannot invent a sentence: every sentence it returns already exists in the index.

## Features

- Sentence-aware chunks with overlap, so a hit does not start in the middle of a thought
- Cosine search over float32 blobs in SQLite, with vectors stored already normalized
- Citations carry character offsets in the source document
- Empty index returns a clear miss instead of crashing
- A dimension mismatch after an embedding-model change returns an explicit error
- Markdown headings stay in the retrieval signal and are stripped from the spoken answer
- Overlapping duplicate sentences are collapsed in the final answer

## Technology Stack

- Python
- FastAPI
- sentence-transformers and PyTorch
- SQLite

## Architecture

`rag/` does not know about HTTP. `rag/api.py` does not know how an embedding is computed. Changing the embedder, the answerer, or the transport touches one file.

| Layer | Default | With `ANTHROPIC_API_KEY` |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local, 384 dimensions | Same |
| Answer | Extractive: the retrieved sentences, unchanged | A hosted model writes prose from those same passages |

The local index is aimed at about 100,000 chunks in one portable SQLite file.

## Installation

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`. Linux or macOS: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
python demo.py
uvicorn rag.api:app --reload
```

The API is at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

## Usage

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Index status and which providers are active |
| `GET` | `/documents` | Indexed documents |
| `POST` | `/documents` | Upload a PDF, Markdown, or text file |
| `DELETE` | `/documents/{id}` | Remove a document and its chunks |
| `POST` | `/query` | Ask a question. Returns the answer and ranked sources |

```bash
curl -F "file=@sample_docs/tenancy-handbook.md" http://localhost:8000/documents

curl -X POST -H "Content-Type: application/json" \
     -d '{"question":"how much is the deposit?","limit":3}' \
     http://localhost:8000/query
```

Each source includes a citation label, document title, chunk number, cosine score, and a 280-character excerpt.

On the bundled sample (3 documents, 4 chunks) the local model on CPU retrieved the deposit, emergency-repair, pet, and refund passages. The pet question does not share the words "third floor" with the corpus. Retrieval still lands on the pet clause.

## Testing

```bash
pytest tests -q
```

The suite has 48 tests.

## Limitations

The extractive answerer will not paraphrase. The hosted answerer is optional and is skipped when the key is missing or the provider returns an empty body. This is not a general web search.

## License

MIT
