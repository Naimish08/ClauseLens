"""Smoke test for the error-analysis report generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.error_analysis import main
from src.data import BENCHMARKS_DIR

requires_data = pytest.mark.skipif(
    not (BENCHMARKS_DIR / "cuad.json").exists(), reason="run `make data` first"
)


@requires_data
def test_error_analysis_writes_grouped_failures(tmp_path: Path) -> None:
    out = tmp_path / "errors.md"
    code = main(
        [
            "--config",
            "configs/baseline.yaml",
            "--limit",
            "8",
            "--per-cause",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "# Error analysis" in text
    assert "| cause | queries |" in text
    assert "**query**" in text
    assert "**expected**" in text
    assert "**returned**" in text
