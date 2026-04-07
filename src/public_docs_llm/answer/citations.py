from __future__ import annotations

from typing import Iterable

from public_docs_llm.models import Citation, RetrievedChunk


def build_citations(chunks: Iterable[RetrievedChunk]) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        citations.append(
            Citation(
                title=chunk.title,
                source_url=chunk.source_url,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                snippet=chunk.snippet,
            )
        )
    return citations

