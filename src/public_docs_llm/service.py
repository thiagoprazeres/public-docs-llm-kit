from __future__ import annotations

from pathlib import Path

from public_docs_llm.answer.citations import build_citations
from public_docs_llm.models import QueryResponse, RetrievedChunk
from public_docs_llm.openai_support import OpenAIService
from public_docs_llm.retrieval import LocalIndexSearcher
from public_docs_llm.settings import AppConfig


def insufficient_evidence_text(citations: list[str]) -> str:
    if citations:
        joined = " ".join(citations)
        return f"I do not have enough grounded evidence in the indexed corpus to answer confidently. {joined}".strip()
    return "I do not have enough grounded evidence in the indexed corpus to answer confidently."


def ensure_citation_markers(answer: str, citation_count: int) -> str:
    if citation_count == 0:
        return answer.strip()
    if "[" in answer and "]" in answer:
        return answer.strip()
    return f"{answer.strip()} [1]"


class GroundedQueryService:
    def __init__(
        self,
        *,
        index_path: Path | str,
        config: AppConfig,
        openai_service: OpenAIService,
        searcher: LocalIndexSearcher | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.config = config
        self.openai_service = openai_service
        self.searcher = searcher or LocalIndexSearcher(
            index_path=self.index_path,
            config=config,
            openai_service=openai_service,
        )

    def _is_weak(self, retrieved_chunks: list[RetrievedChunk]) -> bool:
        if not retrieved_chunks:
            return True
        return retrieved_chunks[0].similarity < self.config.retrieval.weak_evidence_threshold

    def answer(self, question: str, *, top_k: int | None = None) -> QueryResponse:
        retrieved_chunks = self.searcher.search(question, top_k=top_k)
        citations = build_citations(retrieved_chunks)
        weak_evidence = self._is_weak(retrieved_chunks)

        if not retrieved_chunks:
            answer = insufficient_evidence_text([])
        else:
            answer = self.openai_service.answer_question(
                question=question,
                retrieved_chunks=retrieved_chunks,
                model=self.config.models.answer_model,
                max_output_tokens=self.config.models.answer_max_output_tokens,
                weak_evidence=weak_evidence,
            )
            answer = ensure_citation_markers(answer, len(citations))
            if weak_evidence:
                answer = ensure_citation_markers(
                    insufficient_evidence_text([f"[{idx}]" for idx in range(1, len(citations) + 1)])
                    if "enough grounded evidence" in answer.lower()
                    else answer,
                    len(citations),
                )

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            index_name=self.index_path.name,
        )
