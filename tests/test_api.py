from fastapi.testclient import TestClient

from public_docs_llm.api.app import create_app
from public_docs_llm.models import QueryResponse, RetrievedChunk
from public_docs_llm.service import GroundedQueryService
from public_docs_llm.settings import AppConfig


class FakeSearcher:
    def search(self, question: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                title="Help Article",
                source_url="https://help.openai.com/example",
                chunk_id="help::chunk-0000",
                document_id="help-doc",
                snippet="Support info",
                text="Support info text",
                source_type="help_center",
                trust_tier="high",
                similarity=0.18,
                score=0.33,
            )
        ]


class FakeOpenAIService:
    def answer_question(self, **kwargs) -> str:
        return "I do not have enough grounded evidence in the indexed corpus to answer confidently. [1]"


def test_query_endpoint_returns_grounded_answer_with_citations() -> None:
    service = GroundedQueryService(
        index_path="/tmp/openai-public-docs",  # type: ignore[arg-type]
        config=AppConfig(),
        openai_service=FakeOpenAIService(),  # type: ignore[arg-type]
        searcher=FakeSearcher(),  # type: ignore[arg-type]
    )
    app = create_app(service=service)
    client = TestClient(app)

    response = client.post("/query", json={"question": "How do I contact support?"})

    assert response.status_code == 200
    body = QueryResponse.model_validate(response.json())
    assert "[1]" in body.answer
    assert body.citations[0].chunk_id == "help::chunk-0000"
    assert body.retrieved_chunks[0].source_type == "help_center"
