.PHONY: install reproduce smoke test lint format-check security wheel-smoke check

install:
	uv sync --locked --all-extras --dev

reproduce:
	MPLCONFIGDIR=.matplotlib uv run relation-detection reproduce

smoke:
	MPLCONFIGDIR=.matplotlib uv run relation-detection smoke

test:
	MPLCONFIGDIR=.matplotlib uv run pytest

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

security:
	uv run python scripts/check_sensitive.py

wheel-smoke:
	uv build --wheel
	MPLCONFIGDIR=.matplotlib uv run --isolated --no-project --with dist/*.whl relation-detection smoke

check: lint format-check test security
