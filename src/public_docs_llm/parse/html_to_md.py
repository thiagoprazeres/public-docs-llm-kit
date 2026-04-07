from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as html_to_markdown

from public_docs_llm.models import CrawlArtifact, DocumentRecord


NOISY_TAGS = {"script", "style", "noscript", "iframe", "svg", "form", "button"}
NOISY_HINTS = (
    "cookie",
    "footer",
    "header",
    "nav",
    "breadcrumb",
    "sidebar",
    "menu",
    "modal",
    "banner",
)


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def _looks_noisy(tag: Tag) -> bool:
    if tag.name in {"nav", "footer", "aside"}:
        return True
    joined = " ".join(
        [
            tag.get("id", ""),
            " ".join(tag.get("class", [])),
            tag.get("role", ""),
            tag.get("aria-label", ""),
        ]
    ).lower()
    return any(hint in joined for hint in NOISY_HINTS)


def _extract_content_root(soup: BeautifulSoup) -> Tag:
    for selector in ("article", "main", '[role="main"]'):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            return node
    if soup.body:
        return soup.body
    return soup


def normalize_html_artifact(artifact: CrawlArtifact, raw_artifact_path: Path) -> DocumentRecord:
    soup = BeautifulSoup(artifact.html, "html.parser")

    for tag_name in NOISY_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()

    root = _extract_content_root(soup)
    for node in list(root.find_all(True)):
        if isinstance(node, Tag) and _looks_noisy(node):
            node.decompose()

    title = (
        artifact.title_hint
        or (soup.title.string.strip() if soup.title and soup.title.string else None)
        or artifact.document_id
    )
    markdown = html_to_markdown(str(root), heading_style="ATX", bullets="-")
    markdown = _clean_text(markdown)

    return DocumentRecord(
        document_id=artifact.document_id,
        title=title,
        source_url=artifact.url,
        markdown=markdown,
        fetched_at=artifact.fetched_at,
        content_hash=artifact.content_hash,
        fetch_method=artifact.fetch_method,
        source_type=artifact.source_type,
        trust_tier=artifact.trust_tier,
        raw_artifact_path=str(raw_artifact_path),
    )

