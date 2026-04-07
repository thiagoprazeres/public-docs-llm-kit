from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SeedUrl(BaseModel):
    url: str
    source_hint: str | None = None


class CrawlConfig(BaseModel):
    seeds: list[SeedUrl] = Field(default_factory=list)
    user_agent: str = "public-docs-llm-kit/0.1"
    max_pages: int = 10
    request_timeout_seconds: float = 20.0
    same_host_only: bool = True
    browser_fallback_hosts: list[str] = Field(default_factory=lambda: ["help.openai.com"])
    follow_links: bool = True


class StorageConfig(BaseModel):
    corpus_root: str = "var/corpus"
    index_root: str = "var/indexes"


class IndexingConfig(BaseModel):
    index_name: str = "openai-public-docs"
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    snippet_chars: int = 280
    exclude_academy_partner_content: bool = True


class RetrievalConfig(BaseModel):
    top_k: int = 5
    weak_evidence_threshold: float = 0.30
    min_similarity_threshold: float = 0.15
    source_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "help_center": 0.15,
            "academy_official": 0.05,
            "academy_partner": -0.10,
            "other": 0.0,
        }
    )


class ModelConfig(BaseModel):
    embedding_model: str = "text-embedding-3-small"
    answer_model: str = "gpt-5-mini"
    answer_max_output_tokens: int = 400


class AppConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)

    def resolve_path(self, path_value: str | Path, *, base_dir: Path | None = None) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        root = base_dir or Path.cwd()
        return (root / path).resolve()


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PUBLIC_DOCS_LLM_",
        env_file=".env",
        extra="ignore",
    )

    openai_api_key: str | None = None
    default_config_path: str = "configs/indexing.example.yaml"
    default_index_path: str | None = None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of config file: {path}")
    return data


def load_config(config_path: str | Path | None = None) -> tuple[RuntimeSettings, AppConfig, Path]:
    runtime = RuntimeSettings()
    config_file = Path(config_path or runtime.default_config_path).resolve()
    config = AppConfig.model_validate(_read_yaml(config_file))
    base_dir = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    return runtime, config, base_dir


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_or_default(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
