#!/usr/bin/env python3
"""Validate the frozen unanswerable set.

    python scripts/validate_negatives.py
    python scripts/validate_negatives.py --path data/unanswerable.jsonl

Checks, per entry: schema, unique id, type in 1-4, split matches `assign_split`,
the claimed absence still holds against the real contract text, and (type 4) the
near-miss neighbour really exists. Also flags exact duplicates of benchmark
queries, which would make a "negative" answerable by definition.

Exit code 1 on any violation, so this can gate a commit.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_benchmark, load_mini_cuad  # noqa: E402
from src.negatives import (  # noqa: E402
    TYPE_DESCRIPTIONS,
    load_negatives,
    validate_negatives,
)

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "unanswerable.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--show", action="store_true", help="print every entry")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 1

    entries = load_negatives(args.path)
    _, documents = load_mini_cuad()
    report = validate_negatives(entries, documents)

    benchmark = load_benchmark("cuad")
    benchmark_texts = {query.text.strip() for query in benchmark}
    duplicates = [entry.id for entry in entries if entry.query.strip() in benchmark_texts]

    print(f"file: {args.path}")
    print(report.summary())
    for type_id in sorted(report.per_type):
        print(f"  type {type_id}: {report.per_type[type_id]:2d}  {TYPE_DESCRIPTIONS[type_id]}")

    if args.show:
        print()
        for entry in entries:
            scope = "corpus" if entry.scope == "corpus" else entry.scope.split("/")[-1][:45]
            print(f"[{entry.id}] type={entry.type} split={entry.split} scope={scope}")
            print(f"    {entry.query}")
            print(f"    why: {entry.rationale}")

    if duplicates:
        print(f"\nexact duplicates of benchmark queries: {duplicates}", file=sys.stderr)

    for violation in report.violations:
        print(f"VIOLATION {violation}", file=sys.stderr)

    if report.violations or duplicates:
        return 1

    counts = Counter(entry.type for entry in entries)
    if set(counts) != {1, 2, 3, 4}:
        print("VIOLATION: every negative type must be represented", file=sys.stderr)
        return 1
    if not report.per_split.get("train") or not report.per_split.get("test"):
        print("VIOLATION: both train and test splits must be non-empty", file=sys.stderr)
        return 1

    print("\nOK: every absence claim holds and the set is consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
