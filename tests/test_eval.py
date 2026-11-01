"""Tests for the evaluation harness and its guard rails."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from eval import run_eval
from src.config import parse_config
from src.data import Document, Query, Snippet


def _queries() -> list[Query]:
    return [
        Query(
            text="Consider the A Agreement; What is the governing law?",
            snippets=(Snippet("cuad/a.txt", 0, 13, ""),),
        ),
        Query(
            text="Consider the B Agreement; What is the retention period?",
            snippets=(Snippet("cuad/b.txt", 0, 9, ""),),
        ),
    ]


def _documents() -> dict[str, Document]:
    # Deliberately distinct vocabulary so document identification is unambiguous;
    # repeated filler would let stop-word frequency dominate BM25.
    return {
        "cuad/a.txt": Document("cuad/a.txt", "governing law delaware"),
        "cuad/b.txt": Document("cuad/b.txt", "retention period seven years"),
    }


def test_test_split_requires_the_final_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_eval.main(["--config", "configs/baseline.yaml", "--split", "test"])
    assert code == 2
    assert "i-know-this-is-final" in capsys.readouterr().err


def test_evaluate_produces_aggregates(baseline_dict: dict[str, Any]) -> None:
    config = parse_config(baseline_dict)
    result = run_eval.evaluate(config, _queries(), _documents(), ks=(1, 2), mode="paper")

    assert result.aggregates[1].n == 2
    assert result.aggregates[1].doc_hit == 1.0
    assert result.aggregates[2].recall >= result.aggregates[1].recall
    assert len(result.latencies_ms) == 2
    assert result.index_stats["documents"] == 2


def test_write_artefacts_creates_the_expected_files(
    baseline_dict: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_eval, "ARTIFACTS_DIR", tmp_path / "runs")
    config = parse_config(baseline_dict)
    result = run_eval.evaluate(config, _queries(), _documents(), ks=(1, 2))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.as_dict()), encoding="utf-8")
    run_eval.write_artefacts(result, config_path)

    run_dir = result.run_dir
    for name in ("config.yaml", "per_query.jsonl", "aggregate.json", "table.md", "table.csv"):
        assert (run_dir / name).exists(), name

    lines = (run_dir / "per_query.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert set(record) == {"query", "category", "split", "ks"}
    assert "1" in record["ks"]

    table = (run_dir / "table.md").read_text(encoding="utf-8")
    assert "| k |" in table and "recall" in table


def test_table_marks_the_metric_mode(baseline_dict: dict[str, Any]) -> None:
    config = parse_config(baseline_dict)
    result = run_eval.evaluate(config, _queries(), _documents(), ks=(1,), mode="union")
    assert "`union`" in run_eval.render_markdown(result)


def test_ks_include_the_serving_top_k(baseline_dict: dict[str, Any]) -> None:
    config = parse_config(baseline_dict)
    result = run_eval.evaluate(config, _queries(), _documents(), ks=(1, 2))
    assert config.top_k in result.ks


def test_log_test_run_appends_jsonl(
    baseline_dict: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "test_runs.log"
    monkeypatch.setattr(run_eval, "TEST_RUN_LOG", log_path)
    config = parse_config(baseline_dict)
    result = run_eval.evaluate(config, _queries(), _documents(), ks=(1,))

    run_eval.log_test_run(result, "unit test")
    run_eval.log_test_run(result, "unit test again")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[1])
    assert entry["reason"] == "unit test again"
    assert "git_sha" in entry
