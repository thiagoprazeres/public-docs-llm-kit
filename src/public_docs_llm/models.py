from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CrawlArtifact(BaseModel):
    document_id: str
    url: str
    fetched_at: str
    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)
    content_hash: str
    fetch_method: str
    source_type: str
    trust_tier: str
    partner_detected: bool = False
    title_hint: str | None = None
    discovered_links: list[str] = Field(default_factory=list)
    html: str


class DocumentRecord(BaseModel):
    document_id: str
    title: str
    source_url: str
    markdown: str
    fetched_at: str
    content_hash: str
    fetch_method: str
    source_type: str
    trust_tier: str
    raw_artifact_path: str


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source_url: str
    snippet: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    source_type: str
    trust_tier: str
    embedding: list[float] = Field(default_factory=list)


class Citation(BaseModel):
    title: str
    source_url: str
    chunk_id: str
    document_id: str
    snippet: str


class RetrievedChunk(Citation):
    text: str
    source_type: str
    trust_tier: str
    similarity: float
    score: float


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    index_path: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    index_name: str


class IndexMetadata(BaseModel):
    index_name: str
    built_at: str
    document_count: int
    chunk_count: int
    embedding_model: str
    answer_model: str
    chunk_size_chars: int
    chunk_overlap_chars: int
    source_counts: dict[str, int]
    config: dict[str, Any] = Field(default_factory=dict)

