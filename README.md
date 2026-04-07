# Grounded Public Docs LLM Kit

This repository contains a production-oriented MVP for a grounded documentation pipeline over public docs. It crawls a small seeded corpus, preserves provenance, normalizes HTML to Markdown, chunks and embeds documents locally, then answers questions through a FastAPI endpoint with structured citations.

The current vertical slice targets:

- `https://help.openai.com`
- `https://academy.openai.com`

Design priorities:

- correctness over breadth
- provenance everywhere
- citations in every answer
- Help Center ranked above Academy
- Academy partner content excluded by default when detected

This repo does **not** claim direct ingestion from public URLs into ChatGPT Projects. The local corpus/index are designed so they can later feed an external synchronization workflow for curated content in ChatGPT Projects.

## Project layout

- `src/public_docs_llm/`: crawler, parser, indexer, retrieval path, API, CLI
- `scripts/`: thin wrappers for crawl and index build
- `configs/indexing.example.yaml`: runnable example config
- `docs/architecture.md`: compact system design notes
- `var/`: generated crawl and index artifacts (gitignored)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m playwright install chromium
export OPENAI_API_KEY=YOUR_KEY
```

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/uvicorn public_docs_llm.api.app:app --reload
```

You can also run the local validation path with:

```bash
make ci
```

## End-to-end workflow

1. Crawl a small seeded corpus:

```bash
python scripts/crawl_site.py --config configs/indexing.example.yaml --max-pages 10
```

This writes raw crawl artifacts under `var/corpus/<run>/raw/`. Each raw document JSON preserves:

- URL
- fetch timestamp
- HTTP status
- selected response headers
- content hash
- fetch method (`httpx` or `playwright`)
- source classification
- raw HTML

2. Build the local index:

```bash
python scripts/build_index.py --config configs/indexing.example.yaml
```

This writes:

- `var/indexes/openai-public-docs/documents.json`
- `var/indexes/openai-public-docs/chunks.json`
- `var/indexes/openai-public-docs/index.json`

3. Start the API:

```bash
export PYTHONPATH=src
uvicorn public_docs_llm.api.app:app --reload
```

4. Query it:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"How do I contact support?","top_k":5}'
```

You can also query directly from the CLI:

```bash
python -m public_docs_llm.cli query \
  --config configs/indexing.example.yaml \
  --index var/indexes/openai-public-docs \
  --question "How do I contact support?"
```

## API behavior

`POST /query` returns:

- `answer`: grounded answer text with citation markers like `[1]`
- `citations`: structured citations including `title`, `source_url`, `chunk_id`, `document_id`, `snippet`
- `retrieved_chunks`: ranked evidence used for synthesis
- `index_name`: active local index name

Example response shape:

```json
{
  "answer": "OpenAI Help Center directs users to contact support through the Help Center support flow rather than by a public email address. If the in-product option is unavailable, use the Help Center request path described in the article. [1]",
  "citations": [
    {
      "title": "How can I contact support?",
      "source_url": "https://help.openai.com/en/articles/6614161-how-can-i-contact-support",
      "chunk_id": "6614161-how-can-i-contact-support-ef7cb7db49::chunk-0000",
      "document_id": "6614161-how-can-i-contact-support-ef7cb7db49",
      "snippet": "Contact support through the Help Center support flow. The article explains where to submit a request and what details help support investigate issues faster."
    }
  ],
  "retrieved_chunks": [
    {
      "title": "How can I contact support?",
      "source_url": "https://help.openai.com/en/articles/6614161-how-can-i-contact-support",
      "chunk_id": "6614161-how-can-i-contact-support-ef7cb7db49::chunk-0000",
      "document_id": "6614161-how-can-i-contact-support-ef7cb7db49",
      "snippet": "Contact support through the Help Center support flow. The article explains where to submit a request and what details help support investigate issues faster.",
      "text": "Contact support through the Help Center support flow...",
      "source_type": "help_center",
      "trust_tier": "high",
      "similarity": 0.84,
      "score": 0.99
    }
  ],
  "index_name": "openai-public-docs"
}
```

This is an illustrative example, not a guaranteed live response.

The answer path is conservative:

- it only uses retrieved evidence
- it returns an insufficient-evidence answer when retrieval is weak
- it re-ranks with trust bias: Help Center > Academy official > Academy partner/other

## Notes and limitations

- Help Center pages may require the Playwright fallback in environments where plain HTTP fetches receive `403`.
- The crawler is intentionally seed-driven for the MVP. It is not a general crawler framework.
- Academy partner content is excluded by default when the page includes the partner disclaimer text.
- Retrieval is local and file-based today. There is no external vector database in this MVP.

## Tests

```bash
pytest
```
