from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import uvicorn

from public_docs_llm.api.app import create_app, resolve_index_path
from public_docs_llm.crawl import crawl_site
from public_docs_llm.indexing import build_local_index
from public_docs_llm.openai_support import OpenAIService
from public_docs_llm.service import GroundedQueryService
from public_docs_llm.settings import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grounded public docs pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser("crawl", help="Crawl configured public documentation seeds.")
    crawl_parser.add_argument("--config", default=None)
    crawl_parser.add_argument("--max-pages", type=int, default=None)

    build_parser_cmd = subparsers.add_parser("build-index", help="Build a local embedding index.")
    build_parser_cmd.add_argument("--config", default=None)
    build_parser_cmd.add_argument("--run-dir", default=None)

    serve_parser = subparsers.add_parser("serve", help="Run the query API.")
    serve_parser.add_argument("--config", default=None)
    serve_parser.add_argument("--index", default=None)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    query_parser = subparsers.add_parser("query", help="Query a local index directly.")
    query_parser.add_argument("--config", default=None)
    query_parser.add_argument("--index", default=None)
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--top-k", type=int, default=None)

    return parser


def _build_service(config_path: str | None, index_override: str | None) -> GroundedQueryService:
    runtime, config, base_dir = load_config(config_path)
    index_path = resolve_index_path(index_override, runtime=runtime, config=config, base_dir=base_dir)
    return GroundedQueryService(
        index_path=index_path,
        config=config,
        openai_service=OpenAIService(api_key=runtime.openai_api_key),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "crawl":
        _, config, base_dir = load_config(args.config)
        run_dir = crawl_site(config, base_dir=base_dir, max_pages_override=args.max_pages)
        print(run_dir)
        return 0

    if args.command == "build-index":
        runtime, config, base_dir = load_config(args.config)
        run_dir = Path(args.run_dir).resolve() if args.run_dir else None
        index_dir = build_local_index(runtime, config, base_dir=base_dir, run_dir=run_dir)
        print(index_dir)
        return 0

    if args.command == "serve":
        runtime, config, base_dir = load_config(args.config)
        index_path = resolve_index_path(args.index, runtime=runtime, config=config, base_dir=base_dir)
        os.environ["PUBLIC_DOCS_LLM_DEFAULT_INDEX_PATH"] = str(index_path)
        if args.config:
            os.environ["PUBLIC_DOCS_LLM_DEFAULT_CONFIG_PATH"] = args.config
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "query":
        service = _build_service(args.config, args.index)
        response = service.answer(args.question, top_k=args.top_k)
        print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

