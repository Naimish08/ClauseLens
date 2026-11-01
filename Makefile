PY := .venv/bin/python
UV := uv

.DEFAULT_GOAL := help

.PHONY: help setup data verify test lint eval loop app cli clean

help:
	@echo "setup   create .venv (Python 3.12) and install core deps"
	@echo "data    download + unpack LegalBench-RAG into data/"
	@echo "verify  verify ground-truth character offsets"
	@echo "test    run the test suite"
	@echo "lint    run ruff"
	@echo "eval    evaluate a config: make eval CONFIG=configs/baseline.yaml"
	@echo "loop    run the self-improvement loop"
	@echo "app     launch the Streamlit app"
	@echo "cli     run the CLI (ARGS='...' to pass arguments)"

setup:
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"

data:
	$(PY) scripts/download_data.py

verify:
	$(PY) scripts/verify_data.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

eval:
	@test -n "$(CONFIG)" || (echo "CONFIG is required, e.g. CONFIG=configs/baseline.yaml"; exit 1)
	$(PY) -m eval.run_eval --config $(CONFIG)

loop:
	$(PY) -m loop.run_loop --config $(or $(CONFIG),configs/baseline.yaml)

app:
	$(PY) -m streamlit run app/streamlit_app.py

cli:
	$(PY) -m app.cli $(ARGS)

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
