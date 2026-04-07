from public_docs_llm.indexing import chunk_document
from public_docs_llm.models import DocumentRecord
from public_docs_llm.settings import AppConfig


def test_chunk_document_is_deterministic_and_preserves_provenance() -> None:
    config = AppConfig()
    config.indexing.chunk_size_chars = 90
    config.indexing.chunk_overlap_chars = 20

    document = DocumentRecord(
        document_id="doc-123",
        title="Support Article",
        source_url="https://help.openai.com/en/articles/example",
        markdown=(
            "# Support\n\n"
            "First paragraph with enough text to force chunking across multiple blocks.\n\n"
            "Second paragraph keeps the flow going and should appear in a later chunk.\n\n"
            "Third paragraph closes the example."
        ),
        fetched_at="2026-04-07T00:00:00Z",
        content_hash="abc",
        fetch_method="httpx",
        source_type="help_center",
        trust_tier="high",
        raw_artifact_path="/tmp/raw.json",
    )

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].document_id == "doc-123"
    assert first[0].source_type == "help_center"
    assert len(first) >= 2
    assert first[0].start_char <= first[0].end_char
    assert first[1].start_char < first[1].end_char
    assert "Second paragraph" in "\n".join(chunk.text for chunk in first)

