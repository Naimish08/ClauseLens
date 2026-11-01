"""Tests for the CLI and a smoke test for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cli import main as cli_main
from src.data import BENCHMARKS_DIR

requires_data = pytest.mark.skipif(
    not (BENCHMARKS_DIR / "cuad.json").exists(), reason="run `make data` first"
)

MATCHING_QUERY = "What is the governing law of this contract?"


@requires_data
def test_cli_returns_passages_with_offsets(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(["--config", "configs/baseline.yaml", "--query", MATCHING_QUERY])
    assert code == 0
    output = capsys.readouterr().out
    assert "score=" in output
    assert ".txt" in output
    assert "[" in output and "]" in output


@requires_data
def test_cli_abstains_with_an_explanation(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(
        ["--config", "configs/baseline.yaml", "--query", "zzzz qqqq unmatched token"]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "No passage is relevant enough" in output
    assert "reason:" in output


@requires_data
def test_cli_json_output_is_valid(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(
        ["--config", "configs/baseline.yaml", "--query", MATCHING_QUERY, "--json", "--top-k", "3"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == MATCHING_QUERY
    assert "abstained" in payload and "spans" in payload
    for span in payload["spans"]:
        assert {"file_path", "start", "end", "score", "excerpt"} <= set(span)


@requires_data
def test_cli_respects_top_k(capsys: pytest.CaptureFixture[str]) -> None:
    cli_main(
        ["--config", "configs/baseline.yaml", "--query", MATCHING_QUERY, "--json", "--top-k", "2"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["spans"]) <= 2


@requires_data
def test_streamlit_app_renders(capsys: pytest.CaptureFixture[str]) -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60)
    app.run()
    assert not app.exception
    # With no query entered the app shows the prompt, not results.
    assert app.info or app.text_input
