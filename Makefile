ci:
	.venv/bin/python -m compileall src
	.venv/bin/ruff check .
	.venv/bin/pytest -q

