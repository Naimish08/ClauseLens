"""Tests for the experiment-log renderer and validator."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.log_report import load_log, render, validate


def record(
    iteration: int,
    *,
    accepted: bool,
    recall: float = 0.3,
    delta: float | None = None,
    diff: dict | None = None,
    trigger: str = "wrong_section",
    gates: dict | None = None,
) -> dict:
    return {
        "iteration": iteration,
        "accepted": accepted,
        "proposal": {
            "proposer": "baseline" if iteration == 0 else "rule:size_up",
            "trigger": trigger,
            "diff": diff or {},
        },
        "evaluation": {"primary": recall, "primary_delta": delta},
        "decision": gates or {"stage": "baseline"},
    }


def test_render_marks_baseline_and_changes() -> None:
    records = [
        record(0, accepted=True),
        record(
            1,
            accepted=True,
            recall=0.35,
            delta=0.05,
            diff={"chunking.params.size": {"from": 512, "to": 1024}},
            gates={
                "gate_effect_size": True,
                "gate_significance": True,
                "gate_consistency": True,
                "gate_guardrails": True,
                "notes": "accepted",
            },
        ),
    ]
    table = render(records)
    assert "| # | change |" in table
    assert "baseline" in table
    assert "size: 512 → 1024" in table
    assert "+0.050" in table
    assert "accepted" in table


def test_validate_accepts_a_wellformed_log() -> None:
    records = [record(0, accepted=True), record(1, accepted=False)]
    assert validate(records) == []


def test_validate_flags_non_contiguous_iterations() -> None:
    records = [record(0, accepted=True), record(2, accepted=False)]
    assert any("contiguous" in problem for problem in validate(records))


def test_validate_flags_accepted_with_empty_diff() -> None:
    records = [record(0, accepted=True), record(1, accepted=True)]
    assert any("empty diff" in problem for problem in validate(records))


def test_validate_flags_accepted_with_failing_gate() -> None:
    records = [
        record(0, accepted=True),
        record(
            1,
            accepted=True,
            diff={"chunking.params.size": {"from": 1, "to": 2}},
            gates={
                "gate_effect_size": True,
                "gate_significance": False,
                "gate_consistency": True,
                "gate_guardrails": True,
            },
        ),
    ]
    assert any("failing gate" in problem for problem in validate(records))


def test_load_log_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text(
        "\n".join(json.dumps(record(i, accepted=i == 0)) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    assert [r["iteration"] for r in load_log(path)] == [0, 1, 2]


def test_load_log_rejects_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    try:
        load_log(path)
    except ValueError as exc:
        assert "not valid JSON" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_load_log_missing_file(tmp_path: Path) -> None:
    try:
        load_log(tmp_path / "absent.jsonl")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")
