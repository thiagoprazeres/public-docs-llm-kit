from __future__ import annotations

import os
from itertools import islice
from typing import Iterable, Sequence

from openai import OpenAI

from public_docs_llm.models import RetrievedChunk


def batched(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def build_answer_instructions() -> str:
    return (
        "You answer questions only from supplied evidence chunks.\n"
        "Rules:\n"
        "- Do not use outside knowledge.\n"
        "- If the evidence is incomplete or weak, say so plainly.\n"
        "- Cite factual claims inline with bracketed citation markers like [1] or [2].\n"
        "- Do not invent policies, dates, steps, or product behavior.\n"
        "- Prefer Help Center evidence over Academy when the evidence conflicts.\n"
        "- Keep the answer concise and directly responsive."
    )


def build_answer_input(question: str, retrieved_chunks: Sequence[RetrievedChunk], *, weak_evidence: bool) -> str:
    evidence_lines: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        evidence_lines.append(
            "\n".join(
                [
                    f"[{index}] title: {chunk.title}",
                    f"[{index}] url: {chunk.source_url}",
                    f"[{index}] source_type: {chunk.source_type}",
                    f"[{index}] snippet: {chunk.snippet}",
                    f"[{index}] text: {chunk.text}",
                ]
            )
        )

    weakness_note = (
        "The retrieved evidence is weak or only partially relevant. If needed, say you do not have enough grounded evidence."
        if weak_evidence
        else "The retrieved evidence is relevant. Answer only from it."
    )

    return "\n\n".join(
        [
            f"Question: {question}",
            weakness_note,
            "Evidence:",
            "\n\n".join(evidence_lines) if evidence_lines else "(no evidence)",
        ]
    )


def extract_response_text(response: object) -> str:
    direct_text = getattr(response, "output_text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for entry in content:
            text = getattr(entry, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


class OpenAIService:
    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY or PUBLIC_DOCS_LLM_OPENAI_API_KEY."
            )
        self.client = OpenAI(api_key=resolved_key)

    def embed_texts(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in batched(list(texts), batch_size=32):
            response = self.client.embeddings.create(
                input=batch,
                model=model,
                encoding_format="float",
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def answer_question(
        self,
        *,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        model: str,
        max_output_tokens: int,
        weak_evidence: bool,
    ) -> str:
        response = self.client.responses.create(
            model=model,
            instructions=build_answer_instructions(),
            input=build_answer_input(question, retrieved_chunks, weak_evidence=weak_evidence),
            temperature=0,
            max_output_tokens=max_output_tokens,
        )
        text = extract_response_text(response)
        if not text:
            raise ValueError("OpenAI response did not contain any text output.")
        return text

