PY := .venv/bin/python
UV := uv

.DEFAULT_GOAL := help

.PHONY: help setup data verify negatives test lint eval loop app cli clean

help:
	@echo "setup     create .venv (Python 3.12) and install core deps"
	@echo "data      download + unpack LegalBench-RAG into data/"
	@echo "verify    verify ground-truth character offsets"
	@echo "negatives validate the frozen unanswerable set"
	@echo "test      run the test suite"
	@echo "lint      run ruff"
	@echo "eval      evaluate a config: make eval CONFIG=configs/baseline.yaml"
	@echo "grid      run the chunking grid and write the trade-off plot"
	@echo "abstention compare abstention policies and write the trade-off curve"
	@echo "retrieval-grid  compare retrieval methods with chunking held fixed"
	@echo "errors    dump concrete failures grouped by cause"
	@echo "compare   paired bootstrap between two configs: make compare A=... B=..."
	@echo "log-report  render the iteration table from logs/experiments.jsonl"
	@echo "check     verify required deliverables exist and are consistent"
	@echo "loop      run the self-improvement loop"
	@echo "app       launch the Streamlit app"
	@echo "cli       run the CLI (ARGS='...' to pass arguments)"

setup:
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"

data:
	$(PY) scripts/download_data.py

verify:
	$(PY) scripts/verify_data.py

negatives:
	$(PY) scripts/validate_negatives.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

eval:
	@test -n "$(CONFIG)" || (echo "CONFIG is required, e.g. CONFIG=configs/baseline.yaml"; exit 1)
	$(PY) -m eval.run_eval --config $(CONFIG)

grid:
	$(PY) scripts/run_grid.py --config configs/grid_chunk.yaml

abstention:
	$(PY) -m eval.run_abstention --config $(or $(CONFIG),configs/baseline.yaml)

retrieval-grid:
	$(PY) scripts/run_retrieval_grid.py --config configs/grid_retrieval.yaml

errors:
	$(PY) scripts/error_analysis.py --config $(or $(CONFIG),configs/final.yaml) --per-cause 2

compare:
	$(PY) scripts/compare_configs.py --a $(or $(A),configs/baseline.yaml) --b $(or $(B),configs/final.yaml)

log-report:
	$(PY) scripts/log_report.py

check:
	$(PY) scripts/check_deliverables.py

loop:
	$(PY) -m loop.run_loop --config $(or $(CONFIG),configs/baseline.yaml)

app:
	$(PY) -m streamlit run app/streamlit_app.py

cli:
	$(PY) -m app.cli $(ARGS)

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
