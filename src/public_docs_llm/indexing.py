from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from public_docs_llm.models import ChunkRecord, CrawlArtifact, DocumentRecord, IndexMetadata
from public_docs_llm.openai_support import OpenAIService
from public_docs_llm.parse.html_to_md import normalize_html_artifact
from public_docs_llm.settings import AppConfig, RuntimeSettings, ensure_directory


def read_crawl_artifacts(run_dir: Path) -> list[tuple[Path, CrawlArtifact]]:
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw crawl directory not found: {raw_dir}")
    artifacts: list[tuple[Path, CrawlArtifact]] = []
    for path in sorted(raw_dir.glob("*.json")):
        artifact = CrawlArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        artifacts.append((path, artifact))
    if not artifacts:
        raise ValueError(f"No crawl artifacts found in {raw_dir}")
    return artifacts


def find_latest_run(corpus_root: Path) -> Path:
    candidates = sorted([path for path in corpus_root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No crawl runs found in {corpus_root}")
    return candidates[-1]


def make_snippet(text: str, *, snippet_chars: int) -> str:
    squashed = re.sub(r"\s+", " ", text).strip()
    if len(squashed) <= snippet_chars:
        return squashed
    return squashed[: snippet_chars - 3].rstrip() + "..."


def _split_large_block(text: str, start_offset: int, *, chunk_size_chars: int, chunk_overlap_chars: int) -> list[tuple[str, int, int]]:
    slices: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size_chars)
        piece = text[start:end].strip()
        if piece:
            absolute_start = start_offset + start
            absolute_end = start_offset + end
            slices.append((piece, absolute_start, absolute_end))
        if end >= len(text):
            break
        start = max(end - chunk_overlap_chars, start + 1)
    return slices


def blockify_markdown(
    markdown: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|$)", markdown):
        block = match.group(0).strip()
        if not block:
            continue
        start, end = match.span()
        if len(block) <= chunk_size_chars:
            blocks.append((block, start, end))
            continue
        blocks.extend(
            _split_large_block(
                block,
                start,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
        )
    return blocks


def chunk_document(document: DocumentRecord, config: AppConfig) -> list[ChunkRecord]:
    blocks = blockify_markdown(
        document.markdown,
        chunk_size_chars=config.indexing.chunk_size_chars,
        chunk_overlap_chars=config.indexing.chunk_overlap_chars,
    )
    if not blocks:
        return []

    chunks: list[ChunkRecord] = []
    index = 0
    start_block = 0
    while start_block < len(blocks):
        text_parts: list[str] = []
        chunk_start = blocks[start_block][1]
        chunk_end = blocks[start_block][2]
        end_block = start_block

        while end_block < len(blocks):
            candidate = "\n\n".join(text_parts + [blocks[end_block][0]]).strip()
            if text_parts and len(candidate) > config.indexing.chunk_size_chars:
                break
            text_parts.append(blocks[end_block][0])
            chunk_end = blocks[end_block][2]
            end_block += 1
            if len(candidate) >= config.indexing.chunk_size_chars:
                break

        chunk_text = "\n\n".join(text_parts).strip()
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document.document_id}::chunk-{index:04d}",
                document_id=document.document_id,
                title=document.title,
                source_url=document.source_url,
                snippet=make_snippet(chunk_text, snippet_chars=config.indexing.snippet_chars),
                text=chunk_text,
                chunk_index=index,
                start_char=chunk_start,
                end_char=chunk_end,
                source_type=document.source_type,
                trust_tier=document.trust_tier,
                embedding=[],
            )
        )
        index += 1

        if end_block >= len(blocks):
            break

        overlap_chars = 0
        next_start = end_block
        while next_start > start_block:
            previous_block = blocks[next_start - 1]
            overlap_chars += len(previous_block[0])
            if overlap_chars >= config.indexing.chunk_overlap_chars:
                next_start -= 1
                break
            next_start -= 1
        if next_start == start_block:
            next_start = end_block
        start_block = max(next_start, start_block + 1 if end_block == start_block else next_start)
    return chunks


def build_local_index(
    runtime: RuntimeSettings,
    config: AppConfig,
    *,
    base_dir: Path,
    run_dir: Path | None = None,
) -> Path:
    corpus_root = config.resolve_path(config.storage.corpus_root, base_dir=base_dir)
    resolved_run_dir = run_dir or find_latest_run(corpus_root)
    artifacts = read_crawl_artifacts(resolved_run_dir)

    documents: list[DocumentRecord] = []
    for raw_path, artifact in artifacts:
        if artifact.source_type == "academy_partner" and config.indexing.exclude_academy_partner_content:
            continue
        document = normalize_html_artifact(artifact, raw_path)
        if not document.markdown.strip():
            continue
        documents.append(document)

    if not documents:
        raise ValueError("No normalized documents were available to index.")

    chunks: list[ChunkRecord] = []
    for document in documents:
        chunks.extend(chunk_document(document, config))

    if not chunks:
        raise ValueError("Chunking produced no chunks.")

    openai_service = OpenAIService(api_key=runtime.openai_api_key)
    embeddings = openai_service.embed_texts(
        [chunk.text for chunk in chunks],
        model=config.models.embedding_model,
    )
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk.embedding = embedding

    index_root = config.resolve_path(config.storage.index_root, base_dir=base_dir)
    index_dir = ensure_directory(index_root / config.indexing.index_name)

    documents_path = index_dir / "documents.json"
    chunks_path = index_dir / "chunks.json"
    metadata_path = index_dir / "index.json"

    documents_path.write_text(
        json.dumps([doc.model_dump(mode="json") for doc in documents], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    source_counts = Counter(document.source_type for document in documents)
    metadata = IndexMetadata(
        index_name=config.indexing.index_name,
        built_at=datetime.now(UTC).isoformat(),
        document_count=len(documents),
        chunk_count=len(chunks),
        embedding_model=config.models.embedding_model,
        answer_model=config.models.answer_model,
        chunk_size_chars=config.indexing.chunk_size_chars,
        chunk_overlap_chars=config.indexing.chunk_overlap_chars,
        source_counts=dict(source_counts),
        config=config.model_dump(mode="json"),
    )
    metadata_path.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return index_dir

