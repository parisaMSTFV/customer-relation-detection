.PHONY: install reproduce smoke test lint security check

install:
	python -m pip install -e ".[dev]"

reproduce:
	MPLCONFIGDIR=.matplotlib python -m customer_relation_detection.cli reproduce

smoke:
	MPLCONFIGDIR=.matplotlib python -m customer_relation_detection.cli smoke

test:
	MPLCONFIGDIR=.matplotlib python -m pytest

lint:
	python -m ruff check .

security:
	python scripts/check_sensitive.py

check: lint test security
