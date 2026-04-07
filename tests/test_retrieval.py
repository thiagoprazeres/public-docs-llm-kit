import json

from public_docs_llm.answer.citations import build_citations
from public_docs_llm.models import ChunkRecord
from public_docs_llm.retrieval import LocalIndexSearcher
from public_docs_llm.settings import AppConfig


class FakeOpenAIService:
    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        assert model == "text-embedding-3-small"
        return [[1.0, 0.0] for _ in texts]


def test_retrieval_prefers_help_center_and_builds_citations(tmp_path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    chunks = [
        ChunkRecord(
            chunk_id="help::chunk-0000",
            document_id="help-doc",
            title="Help Article",
            source_url="https://help.openai.com/example",
            snippet="Help snippet",
            text="Help center text",
            chunk_index=0,
            start_char=0,
            end_char=20,
            source_type="help_center",
            trust_tier="high",
            embedding=[0.70, 0.30],
        ),
        ChunkRecord(
            chunk_id="academy::chunk-0000",
            document_id="academy-doc",
            title="Academy Article",
            source_url="https://academy.openai.com/example",
            snippet="Academy snippet",
            text="Academy text",
            chunk_index=0,
            start_char=0,
            end_char=20,
            source_type="academy_official",
            trust_tier="medium",
            embedding=[0.74, 0.26],
        ),
    ]
    (index_dir / "chunks.json").write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks]),
        encoding="utf-8",
    )

    config = AppConfig()
    searcher = LocalIndexSearcher(
        index_path=index_dir,
        config=config,
        openai_service=FakeOpenAIService(),
    )

    retrieved = searcher.search("support question", top_k=2)
    citations = build_citations(retrieved)

    assert retrieved[0].source_type == "help_center"
    assert citations[0].chunk_id == "help::chunk-0000"
    assert citations[0].title == "Help Article"

