from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from public_docs_llm.models import CrawlArtifact
from public_docs_llm.settings import AppConfig, ensure_directory


PARTNER_DISCLAIMER_PATTERNS = (
    "trusted partner in ai education",
    "does not necessarily represent openai",
)


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def make_document_id(url: str) -> str:
    parsed = urlsplit(url)
    last_segment = parsed.path.rstrip("/").split("/")[-1] or "index"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{slugify(last_segment)}-{digest}"


def classify_source(url: str, html: str) -> str:
    host = urlsplit(url).netloc.lower()
    normalized_html = html.lower().replace("’", "'")
    if "help.openai.com" in host:
        return "help_center"
    if "academy.openai.com" in host:
        if all(pattern in normalized_html for pattern in PARTNER_DISCLAIMER_PATTERNS):
            return "academy_partner"
        return "academy_official"
    return "other"


def trust_tier_for_source(source_type: str) -> str:
    if source_type == "help_center":
        return "high"
    if source_type == "academy_official":
        return "medium"
    return "low"


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        joined = canonicalize_url(urljoin(base_url, anchor["href"]))
        parsed = urlsplit(joined)
        if parsed.scheme not in {"http", "https"}:
            continue
        links.append(joined)
    seen: set[str] = set()
    ordered: list[str] = []
    for link in links:
        if link in seen:
            continue
        seen.add(link)
        ordered.append(link)
    return ordered


@dataclass
class FetchResult:
    url: str
    status_code: int
    headers: dict[str, str]
    html: str
    fetch_method: str
    title_hint: str | None = None


class PlaywrightBrowserFetcher:
    def __init__(self, *, user_agent: str, timeout_seconds: float) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self._playwright = None
        self._browser = None
        self._context = None

    def _ensure_started(self) -> None:
        if self._context is not None:
            return
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(user_agent=self.user_agent)

    def fetch(self, url: str) -> FetchResult:
        self._ensure_started()
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        page = self._context.new_page()
        try:
            try:
                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=int(self.timeout_seconds * 1000),
                )
            except PlaywrightTimeoutError:
                response = None
            page.wait_for_timeout(1500)
            html = page.content()
            title = page.title() or None
            headers = dict(response.all_headers()) if response is not None else {}
            status_code = response.status if response is not None else 200
            return FetchResult(
                url=url,
                status_code=status_code,
                headers=headers,
                html=html,
                fetch_method="playwright",
                title_hint=title,
            )
        finally:
            page.close()

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None


class RobotsPolicy:
    def __init__(self, client: httpx.Client, *, user_agent: str) -> None:
        self.client = client
        self.user_agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}
        self._crawl_delays: dict[str, float] = {}
        self._last_fetch_at: dict[str, float] = {}

    def _load_host(self, host: str, scheme: str) -> None:
        if host in self._parsers:
            return
        parser = RobotFileParser()
        robots_url = f"{scheme}://{host}/robots.txt"
        try:
            response = self.client.get(robots_url)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser.parse([])
        except httpx.HTTPError:
            parser.parse([])
        self._parsers[host] = parser
        crawl_delay = parser.crawl_delay(self.user_agent) or parser.crawl_delay("*")
        self._crawl_delays[host] = float(crawl_delay or 0.0)

    def wait_if_needed(self, url: str) -> None:
        parsed = urlsplit(url)
        self._load_host(parsed.netloc, parsed.scheme)
        delay = self._crawl_delays.get(parsed.netloc, 0.0)
        if delay <= 0:
            return
        last_seen = self._last_fetch_at.get(parsed.netloc)
        if last_seen is None:
            return
        remaining = delay - (time.monotonic() - last_seen)
        if remaining > 0:
            time.sleep(remaining)

    def mark_fetch(self, url: str) -> None:
        parsed = urlsplit(url)
        self._last_fetch_at[parsed.netloc] = time.monotonic()

    def is_allowed(self, url: str) -> bool:
        parsed = urlsplit(url)
        self._load_host(parsed.netloc, parsed.scheme)
        parser = self._parsers[parsed.netloc]
        return parser.can_fetch(self.user_agent, url)


def _select_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    keys = {"content-type", "etag", "last-modified", "cache-control"}
    selected: dict[str, str] = {}
    items = headers.items() if hasattr(headers, "items") else headers.items()
    for key, value in items:
        if key.lower() in keys:
            selected[key.lower()] = value
    return selected


def _http_fetch(client: httpx.Client, url: str) -> FetchResult:
    response = client.get(url, follow_redirects=True)
    return FetchResult(
        url=str(response.url),
        status_code=response.status_code,
        headers=_select_headers(response.headers),
        html=response.text,
        fetch_method="httpx",
    )


def _is_same_host(seed_hosts: Iterable[str], url: str) -> bool:
    return urlsplit(url).netloc in set(seed_hosts)


def crawl_site(
    config: AppConfig,
    *,
    base_dir: Path,
    max_pages_override: int | None = None,
) -> Path:
    corpus_root = config.resolve_path(config.storage.corpus_root, base_dir=base_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ensure_directory(corpus_root / run_id)
    raw_dir = ensure_directory(run_dir / "raw")

    max_pages = max_pages_override or config.crawl.max_pages
    seed_urls = [canonicalize_url(seed.url) for seed in config.crawl.seeds]
    seed_hosts = {urlsplit(url).netloc for url in seed_urls}
    queue: deque[str] = deque(seed_urls)
    visited: set[str] = set()
    browser_fetcher: PlaywrightBrowserFetcher | None = None

    with httpx.Client(
        headers={"User-Agent": config.crawl.user_agent},
        timeout=config.crawl.request_timeout_seconds,
        follow_redirects=True,
    ) as client:
        robots = RobotsPolicy(client, user_agent=config.crawl.user_agent)
        try:
            while queue and len(visited) < max_pages:
                url = canonicalize_url(queue.popleft())
                if url in visited:
                    continue
                if config.crawl.same_host_only and not _is_same_host(seed_hosts, url):
                    continue
                if not robots.is_allowed(url):
                    visited.add(url)
                    continue

                try:
                    robots.wait_if_needed(url)
                    fetch_result = _http_fetch(client, url)
                    robots.mark_fetch(url)
                except httpx.HTTPError:
                    visited.add(url)
                    continue

                if (
                    fetch_result.status_code in {401, 403}
                    and urlsplit(url).netloc in config.crawl.browser_fallback_hosts
                ):
                    try:
                        browser_fetcher = browser_fetcher or PlaywrightBrowserFetcher(
                            user_agent=config.crawl.user_agent,
                            timeout_seconds=config.crawl.request_timeout_seconds,
                        )
                        fetch_result = browser_fetcher.fetch(url)
                    except Exception:
                        visited.add(url)
                        continue

                visited.add(url)
                if fetch_result.status_code >= 400:
                    continue

                html = fetch_result.html
                source_type = classify_source(fetch_result.url, html)
                trust_tier = trust_tier_for_source(source_type)
                partner_detected = source_type == "academy_partner"
                discovered_links = extract_links(html, fetch_result.url)
                artifact = CrawlArtifact(
                    document_id=make_document_id(fetch_result.url),
                    url=fetch_result.url,
                    fetched_at=datetime.now(UTC).isoformat(),
                    status_code=fetch_result.status_code,
                    headers=fetch_result.headers,
                    content_hash=hashlib.sha256(html.encode("utf-8")).hexdigest(),
                    fetch_method=fetch_result.fetch_method,
                    source_type=source_type,
                    trust_tier=trust_tier,
                    partner_detected=partner_detected,
                    title_hint=fetch_result.title_hint,
                    discovered_links=discovered_links,
                    html=html,
                )
                artifact_path = raw_dir / f"{artifact.document_id}.json"
                artifact_path.write_text(
                    json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                if not config.crawl.follow_links:
                    continue

                for link in discovered_links:
                    canonical = canonicalize_url(link)
                    if canonical in visited:
                        continue
                    if config.crawl.same_host_only and not _is_same_host(seed_hosts, canonical):
                        continue
                    queue.append(canonical)
        finally:
            if browser_fetcher is not None:
                browser_fetcher.close()

    return run_dir
