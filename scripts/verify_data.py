#!/usr/bin/env python3
"""Verify ground-truth character offsets for the CUAD-mini benchmark.

Every downstream metric is a function of character spans, so a silent offset
error would invalidate the whole project.  This script re-derives every span and
fails loudly if anything disagrees.

    python scripts/verify_data.py
    python scripts/verify_data.py --limit 0     # verify the full 4042-query CUAD set
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    ARTIFACTS_DIR,
    MINI_LIMIT,
    load_benchmark,
    load_corpus,
    select_mini,
    split_counts,
    verify_offsets,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--limit", type=int, default=MINI_LIMIT, help="0 = no cap (full CUAD benchmark)"
    )
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "data_verification.json")
    args = parser.parse_args(argv)

    all_queries = load_benchmark("cuad")
    queries = select_mini(all_queries, limit=args.limit) if args.limit else all_queries
    referenced = sorted({fp for q in queries for fp in q.file_paths})

    print(f"benchmark: cuad | loaded {len(all_queries)} queries | evaluating {len(queries)}")
    print(f"documents referenced: {len(referenced)}")

    reports = []
    failures = 0
    for mode in ("raw", "text"):
        documents = load_corpus(referenced, mode=mode)  # type: ignore[arg-type]
        report = verify_offsets(queries, documents, mode=mode)
        print(f"  {report.summary()}")
        for problem in report.problems[:10]:
            print(f"    ! {problem.kind}: {problem.file_path} {problem.span}")
        if report.overlapping_span_queries:
            n_bad = len(report.overlapping_span_queries)
            print(f"    ! {n_bad} queries have non-disjoint GT spans")
        failures += len(report.problems) + len(report.overlapping_span_queries)
        reports.append(
            {
                "mode": report.mode,
                "queries": report.queries,
                "snippets": report.snippets,
                "documents": report.documents,
                "problems": [p.as_dict() for p in report.problems],
                "crlf_documents": list(report.crlf_documents),
                "overlapping_span_queries": list(report.overlapping_span_queries),
            }
        )

    categories = Counter(q.category for q in queries)
    snippets_per_query = Counter(len(q) for q in queries)
    gt_lengths = sorted(s.length for q in queries for s in q.snippets)
    prefix_to_docs: dict[str, set[str]] = {}
    for query in queries:
        prefix_to_docs.setdefault(query.doc_hint, set()).update(query.file_paths)

    profile = {
        "queries": len(queries),
        "documents": len(referenced),
        "snippets": sum(len(q) for q in queries),
        "splits": split_counts(queries),
        "distinct_categories": len(categories),
        "categories": dict(categories.most_common()),
        "snippets_per_query": {str(k): v for k, v in sorted(snippets_per_query.items())},
        "gt_span_chars": {
            "min": gt_lengths[0] if gt_lengths else 0,
            "median": gt_lengths[len(gt_lengths) // 2] if gt_lengths else 0,
            "max": gt_lengths[-1] if gt_lengths else 0,
            "mean": round(sum(gt_lengths) / len(gt_lengths), 1) if gt_lengths else 0,
        },
        "document_hints": len(prefix_to_docs),
        "document_hint_unique": all(len(docs) == 1 for docs in prefix_to_docs.values()),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"reports": reports, "profile": profile}, indent=2) + "\n", encoding="utf-8"
    )

    print("\nprofile")
    print(f"  splits: {profile['splits']}")
    print(f"  snippets: {profile['snippets']} across {len(snippets_per_query)} distinct counts")
    print(f"  categories: {profile['distinct_categories']}")
    print(f"  GT span chars: {profile['gt_span_chars']}")
    print(
        f"  document hints: {profile['document_hints']} "
        f"(1:1 with documents: {profile['document_hint_unique']})"
    )
    print(f"\nwrote {args.out}")

    if failures:
        print(f"\nFAILED: {failures} integrity problem(s)", file=sys.stderr)
        return 1
    print("\nOK: every retrieved span matches its stored answer text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
