from __future__ import annotations

import sys

from public_docs_llm.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["crawl", *sys.argv[1:]]))

