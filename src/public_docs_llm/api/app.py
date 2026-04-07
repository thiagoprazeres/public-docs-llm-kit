from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from public_docs_llm.models import QueryRequest, QueryResponse
from public_docs_llm.openai_support import OpenAIService
from public_docs_llm.service import GroundedQueryService
from public_docs_llm.settings import AppConfig, RuntimeSettings, load_config


def resolve_index_path(
    index_override: str | None,
    *,
    runtime: RuntimeSettings,
    config: AppConfig,
    base_dir: Path,
) -> Path:
    if index_override:
        return Path(index_override).resolve()
    if runtime.default_index_path:
        return Path(runtime.default_index_path).resolve()
    return config.resolve_path(
        f"{config.storage.index_root}/{config.indexing.index_name}",
        base_dir=base_dir,
    )


@lru_cache(maxsize=8)
def load_service(config_path: str | None, index_override: str | None) -> GroundedQueryService:
    runtime, config, base_dir = load_config(config_path)
    index_path = resolve_index_path(index_override, runtime=runtime, config=config, base_dir=base_dir)
    return GroundedQueryService(
        index_path=index_path,
        config=config,
        openai_service=OpenAIService(api_key=runtime.openai_api_key),
    )


def create_app(service: GroundedQueryService | None = None) -> FastAPI:
    app = FastAPI(title="Grounded Public Docs API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        try:
            query_service = service or load_service(None, request.index_path)
            return query_service.answer(request.question, top_k=request.top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive surface for API errors
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()

