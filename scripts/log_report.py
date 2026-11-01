#!/usr/bin/env python3
"""Render the iteration-history table from logs/experiments.jsonl.

    python scripts/log_report.py
    python scripts/log_report.py --validate

The report's iteration table is generated, never transcribed: if the table and the
JSONL disagree, the table is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ARTIFACTS_DIR, REPO_ROOT  # noqa: E402

DEFAULT_LOG = REPO_ROOT / "logs" / "experiments.jsonl"


def load_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"no experiment log at {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return records


def describe_change(record: dict[str, Any]) -> str:
    diff = record.get("proposal", {}).get("diff") or {}
    if not diff:
        return record.get("proposal", {}).get("proposer", "baseline")
    parts = []
    for key, change in list(diff.items())[:2]:
        before, after = change.get("from"), change.get("to")
        if isinstance(after, (dict, list)):
            after = "{…}"
        if isinstance(before, (dict, list)):
            before = "{…}"
        short = key.replace("chunking.params.", "").replace("retrieval.", "")
        parts.append(f"{short}: {before} → {after}")
    return "; ".join(parts)


def render(records: list[dict[str, Any]]) -> str:
    lines = [
        "| # | change | failure addressed | recall@10 | Δ | accepted | why |",
        "|---|---|---|---|---|---|---|",
    ]
    previous = None
    for record in records:
        evaluation = record.get("evaluation", {})
        recall = evaluation.get("primary", 0.0)
        delta = evaluation.get("primary_delta")
        if delta is None:
            delta_text = "—"
        else:
            delta_text = f"{delta:+.3f}"
        accepted = record.get("accepted", False)
        why = record.get("decision", {}).get("notes") or record.get("decision", {}).get("stage", "")
        lines.append(
            f"| {record.get('iteration', '?')} | {describe_change(record)} "
            f"| {record.get('proposal', {}).get('trigger', '')} "
            f"| {recall:.3f} | {delta_text} "
            f"| {'yes' if accepted else 'no'} | {why} |"
        )
        previous = record
    del previous
    return "\n".join(lines)


def validate(records: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    iterations = [record.get("iteration") for record in records]
    if iterations != list(range(len(records))):
        problems.append(f"iterations are not contiguous from 0: {iterations}")

    for record in records:
        iteration = record.get("iteration")
        accepted = record.get("accepted")
        diff = record.get("proposal", {}).get("diff") or {}
        decision = record.get("decision", {})

        if iteration == 0 and accepted is not True:
            problems.append("baseline iteration must be marked accepted")
        if accepted and iteration != 0 and not diff:
            problems.append(f"iteration {iteration} accepted with an empty diff")
        if accepted is False and decision.get("accepted") is True:
            problems.append(f"iteration {iteration} log disagrees with its gate result")
        if accepted is True and iteration != 0:
            gates = (
                decision.get("gate_effect_size"),
                decision.get("gate_significance"),
                decision.get("gate_consistency"),
                decision.get("gate_guardrails"),
            )
            if any(gate is False for gate in gates):
                problems.append(f"iteration {iteration} accepted despite a failing gate")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "iteration_history.md")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    try:
        records = load_log(args.log)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    problems = validate(records)
    for problem in problems:
        print(f"VIOLATION {problem}", file=sys.stderr)

    table = render(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {args.out}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
