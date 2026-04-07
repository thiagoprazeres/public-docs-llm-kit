# Architecture

## Goal

Ship a narrow, inspectable vertical slice for grounded Q&A over public documentation sources with provenance at every stage.

## Flow

1. `crawl_site`
   - starts from explicit seed URLs
   - fetches `robots.txt` per host and honors `crawl-delay` when present
   - fetches pages with `httpx`
   - falls back to Playwright for browser-protected pages like `help.openai.com`
   - writes raw HTML artifacts to `var/corpus/<run>/raw/`

2. `build_index`
   - reads raw crawl artifacts
   - normalizes HTML to Markdown
   - classifies source trust:
     - `help_center`
     - `academy_official`
     - `academy_partner`
   - excludes partner content by default when configured
   - chunks Markdown deterministically with overlap
   - generates embeddings with OpenAI
   - writes `documents.json`, `chunks.json`, and `index.json`

3. `POST /query`
   - embeds the query locally through OpenAI
   - loads chunk embeddings from `chunks.json`
   - computes cosine similarity locally
   - applies trust-weight re-ranking
   - synthesizes an answer from retrieved evidence only
   - returns structured citations

## Provenance model

Every stage preserves enough metadata to inspect how an answer was formed:

- raw artifact: URL, timestamp, status, selected headers, fetch method, source type, raw HTML
- normalized document: document ID, title, source URL, raw artifact path, Markdown
- chunk: chunk ID, document ID, source URL, snippet, character span, source type, embedding
- query result: similarity score, final score, citations, answer text

## Trust rules

- Help Center is highest-trust source.
- OpenAI Academy official content is lower than Help Center.
- Academy partner content is lowest-trust and excluded by default when detectable.

This makes the local pipeline compatible with a later external sync process for curated content into ChatGPT Projects, while avoiding any claim that public URLs can be ingested natively into ChatGPT Projects today.

