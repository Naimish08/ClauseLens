"""Tests for the paired baseline-vs-candidate comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compare_configs import compare, render, vector
from src.data import BENCHMARKS_DIR
from src.metrics import QueryScore

requires_data = pytest.mark.skipif(
    not (BENCHMARKS_DIR / "cuad.json").exists(), reason="run `make data` first"
)


def score(recall: float, precision: float = 0.1) -> QueryScore:
    return QueryScore(
        query="q",
        category="c",
        split="train",
        answerable=True,
        abstained=False,
        k=10,
        n_retrieved=10,
        retrieved_len=1000,
        overlap_len=int(recall * 100),
        gt_len=100,
        recall=recall,
        precision=precision,
        f1=0.0,
        doc_hit=True,
        span_hit=recall > 0,
        chunk_hit=recall >= 1.0,
        mrr=0.5,
    )


def test_vector_uses_answerable_queries_only() -> None:
    unanswerable = QueryScore(
        query="q",
        category="c",
        split="train",
        answerable=False,
        abstained=True,
        k=10,
        n_retrieved=0,
        retrieved_len=0,
        overlap_len=0,
        gt_len=0,
        recall=0.0,
        precision=0.0,
        f1=0.0,
        doc_hit=False,
        span_hit=False,
        chunk_hit=False,
        mrr=0.0,
    )
    assert vector([score(0.5), unanswerable], "recall") == [0.5]


def test_compare_reports_a_clear_difference() -> None:
    rows = compare(
        [score(0.2)] * 20,
        [score(0.5)] * 20,
        k=10,
        seed=13,
        n_resamples=500,
    )
    recall_row = next(row for row in rows if row["metric"] == "recall@10")
    assert recall_row["delta"] == pytest.approx(0.3)
    assert recall_row["significant"] is True
    assert recall_row["delta_ci"][0] > 0


def test_compare_reports_no_difference_for_identical_systems() -> None:
    rows = compare([score(0.3)] * 20, [score(0.3)] * 20, k=10, seed=13, n_resamples=500)
    for row in rows:
        assert row["delta"] == pytest.approx(0.0)
        assert row["significant"] is False


def test_render_includes_intervals_and_significance() -> None:
    rows = compare([score(0.2)] * 10, [score(0.4)] * 10, k=10, seed=13, n_resamples=200)
    table = render(rows, "A", "B")
    assert "| metric | A | B | delta" in table
    assert "recall@10" in table
    assert "yes" in table


@requires_data
def test_cli_compares_two_configs(tmp_path: Path) -> None:
    from scripts.compare_configs import main

    out = tmp_path / "cmp.md"
    code = main(
        [
            "--a",
            "configs/baseline.yaml",
            "--b",
            "configs/baseline.yaml",
            "--limit",
            "8",
            "--resamples",
            "200",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "# Paired comparison" in text
    # Comparing a config with itself must never claim significance.
    assert "| yes |" not in text
