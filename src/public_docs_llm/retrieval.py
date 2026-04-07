from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from public_docs_llm.models import ChunkRecord, RetrievedChunk
from public_docs_llm.openai_support import OpenAIService
from public_docs_llm.settings import AppConfig


@lru_cache(maxsize=8)
def load_chunks(index_path: str) -> tuple[ChunkRecord, ...]:
    chunks_path = Path(index_path) / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Index chunks file not found: {chunks_path}")
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    return tuple(ChunkRecord.model_validate(item) for item in payload)


def _cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    safe_denominator = np.where(matrix_norms * query_norm == 0, 1e-12, matrix_norms * query_norm)
    return (matrix @ query_vector) / safe_denominator


class LocalIndexSearcher:
    def __init__(self, *, index_path: Path, config: AppConfig, openai_service: OpenAIService) -> None:
        self.index_path = index_path
        self.config = config
        self.openai_service = openai_service

    def search(self, question: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        chunks = list(load_chunks(str(self.index_path.resolve())))
        if not chunks:
            return []
        if any(not chunk.embedding for chunk in chunks):
            raise ValueError("Index chunks are missing embeddings.")

        query_embedding = self.openai_service.embed_texts(
            [question],
            model=self.config.models.embedding_model,
        )[0]
        query_vector = np.array(query_embedding, dtype=float)
        embedding_matrix = np.array([chunk.embedding for chunk in chunks], dtype=float)
        similarities = _cosine_similarity(query_vector, embedding_matrix)

        scored: list[RetrievedChunk] = []
        source_weights = self.config.retrieval.source_weights
        for chunk, similarity in zip(chunks, similarities, strict=True):
            similarity_value = float(similarity)
            if similarity_value < self.config.retrieval.min_similarity_threshold:
                continue
            score = similarity_value + source_weights.get(
                chunk.source_type,
                source_weights.get("other", 0.0),
            )
            scored.append(
                RetrievedChunk(
                    title=chunk.title,
                    source_url=chunk.source_url,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    snippet=chunk.snippet,
                    text=chunk.text,
                    source_type=chunk.source_type,
                    trust_tier=chunk.trust_tier,
                    similarity=similarity_value,
                    score=float(score),
                )
            )
        scored.sort(key=lambda item: (item.score, item.similarity), reverse=True)
        limit = top_k or self.config.retrieval.top_k
        return scored[:limit]

