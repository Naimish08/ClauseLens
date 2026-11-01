#!/usr/bin/env python3
"""Check that the required deliverables exist and are internally consistent.

    python scripts/check_deliverables.py

Prints a checklist and exits non-zero if anything required is missing. This is the
pre-submission guard: it catches an empty REPORT section, a missing experiment log
or an unparseable config before a reviewer does.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data import REPO_ROOT  # noqa: E402
from src.negatives import load_negatives, validate_negatives  # noqa: E402

REQUIRED_FILES = (
    "README.md",
    "REPORT.md",
    "AGENTS.md",
    "pyproject.toml",
    "Makefile",
    "configs/baseline.yaml",
    "configs/final.yaml",
    "logs/experiments.jsonl",
    "data/unanswerable.jsonl",
    "docs/README.md",
    "docs/00-overview.md",
    "docs/09-decisions.md",
    "docs/12-risks-and-limitations.md",
    "src/data.py",
    "src/metrics.py",
    "src/pipeline.py",
    "tests",
)

REQUIRED_REPORT_SECTIONS = (
    "## 0. Summary",
    "## 1. System design",
    "## 2. Evaluation setup",
    "## 3. Baseline",
    "## 4. Chunking comparison",
    "## 5. Abstention",
    "## 6. Retrieval comparison",
    "## 7. Iteration history",
    "## 8. Baseline vs final",
    "## 9. Error analysis",
    "## 10. Limitations and next steps",
    "## 11. AI tool usage",
)


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if condition:
            print(f"  ok    {message}")
        else:
            print(f"  FAIL  {message}")
            self.failures.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            print(f"  warn  {message}")
            self.warnings.append(message)
        else:
            print(f"  ok    {message}")


def main() -> int:
    check = Check()
    print("required files")
    for relative in REQUIRED_FILES:
        path = REPO_ROOT / relative
        check.require(path.exists(), relative)

    print("\nconfigs parse")
    for config_path in ("configs/baseline.yaml", "configs/final.yaml"):
        try:
            config = load_config(REPO_ROOT / config_path)
            check.require(True, f"{config_path} -> {config.name}")
        except Exception as exc:  # noqa: BLE001 - report any parse failure
            check.require(False, f"{config_path} ({exc})")

    print("\nexperiment log")
    log_path = REPO_ROOT / "logs" / "experiments.jsonl"
    if log_path.exists():
        import json

        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        check.require(len(lines) >= 2, f"has {len(lines)} iterations (baseline + at least one)")
        check.require(
            any(json.loads(line).get("accepted") is False for line in lines),
            "contains at least one rejected iteration",
        )
        iterations = [json.loads(line)["iteration"] for line in lines]
        check.require(iterations == list(range(len(lines))), "iterations are contiguous from 0")

    print("\nunanswerable set")
    negatives_path = REPO_ROOT / "data" / "unanswerable.jsonl"
    if negatives_path.exists():
        entries = load_negatives(negatives_path)
        check.require(len(entries) >= 30, f"{len(entries)} entries (minimum 30)")
        check.require(
            {entry.type for entry in entries} == {1, 2, 3, 4}, "all four negative types present"
        )
        if (REPO_ROOT / "data" / "benchmarks" / "cuad.json").exists():
            from src.data import load_mini_cuad

            _, documents = load_mini_cuad()
            report = validate_negatives(entries, documents)
            check.require(report.ok, "every absence claim still holds")

    print("\nreport")
    report_path = REPO_ROOT / "REPORT.md"
    if report_path.exists():
        text = report_path.read_text()
        for section in REQUIRED_REPORT_SECTIONS:
            check.require(section in text, f"section present: {section}")
        check.warn(text.count("TODO") == 0, f"{text.count('TODO')} TODO marker(s) remain")

    print("\ntest-run discipline")
    test_log = REPO_ROOT / "logs" / "test_runs.log"
    if test_log.exists():
        runs = [line for line in test_log.read_text().splitlines() if line.strip()]
        check.require(
            len(runs) >= 1,
            f"{len(runs)} test runs recorded (all after tuning; see REPORT.md section 0)",
        )
    else:
        check.warn(False, "test split has not been evaluated yet")

    print()
    if check.failures:
        print(f"{len(check.failures)} required item(s) missing")
        return 1
    if check.warnings:
        print(f"{len(check.warnings)} warning(s)")
    else:
        print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
