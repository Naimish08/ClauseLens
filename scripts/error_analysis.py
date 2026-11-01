#!/usr/bin/env python3
"""Dump concrete retrieval failures, grouped by cause, for the report.

    python scripts/error_analysis.py --config configs/final.yaml --per-cause 2

Each example shows the query, the first ground-truth span, the top retrieved span,
and a one-line diagnosis. The output is markdown so it can be pasted into
``REPORT.md`` and then edited for prose.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data import ARTIFACTS_DIR, load_benchmark_split  # noqa: E402
from src.metrics import score_query  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

DIAGNOSIS = {
    "wrong_document": (
        "The top result is from a different contract: the ranking ignored the "
        "document the question refers to."
    ),
    "wrong_section": (
        "The right contract was found, but the retrieved passage does not overlap "
        "the answer: ranking inside the document put a different clause first."
    ),
    "partial_span": (
        "Retrieval reached the answer but the chunk boundary cut through it, so only "
        "part of the ground truth was returned."
    ),
    "extra_context": (
        "The answer was returned inside a chunk that also carried a large amount of "
        "unrelated text, which costs character precision."
    ),
    "false_abstention": (
        "The system returned nothing for an answerable query."
    ),
    "missed_abstention": (
        "The system returned a passage for an unanswerable query."
    ),
}

CATEGORY_ORDER = tuple(DIAGNOSIS)


def excerpt(text: str, width: int = 240) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= width else collapsed[:width] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--per-cause", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "error_analysis.md")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    split = args.split or config.data.split
    queries, documents = load_benchmark_split(
        benchmark=config.data.benchmark,
        mini_limit=config.data.mini_limit,
        read_mode=config.data.read_mode,
        split=split,
    )
    if args.limit:
        queries = queries[: args.limit]

    pipeline = Pipeline.build(config)
    pipeline.index(documents.values())

    failures: dict[str, list] = defaultdict(list)
    total = 0
    for query in queries:
        result = pipeline.search(query.text, k=args.k)
        score = score_query(query, result.spans, k=args.k)
        total += 1
        if score.failure is not None:
            failures[score.failure].append((score, query, result))

    lines = [
        f"# Error analysis — `{config.name}`, split `{split}`, k={args.k}",
        "",
        f"{sum(len(v) for v in failures.values())} failures out of {total} queries.",
        "",
        "| cause | queries | share of failures |",
        "|---|---|---|",
    ]
    total_failures = sum(len(v) for v in failures.values()) or 1
    for cause in CATEGORY_ORDER:
        count = len(failures.get(cause, []))
        if count:
            lines.append(f"| {cause} | {count} | {count / total_failures:.0%} |")

    for cause in CATEGORY_ORDER:
        examples = failures.get(cause, [])
        if not examples:
            continue
        examples.sort(key=lambda item: item[0].recall)
        lines += ["", f"## {cause} ({len(examples)} queries)", "", f"*{DIAGNOSIS[cause]}*", ""]

        for index, (score, query, result) in enumerate(examples[: args.per_cause], start=1):
            lines.append(f"### {cause} example {index}")
            lines.append("")
            lines.append(f"- **query**: {excerpt(query.text, 300)}")
            snippet = query.snippets[0]
            lines.append(
                f"- **expected**: `{snippet.file_path.split('/')[-1][:60]}` "
                f"[{snippet.start}:{snippet.end}] — {excerpt(snippet.answer, 200)}"
            )
            if result.spans:
                top = result.spans[0]
                text = documents[top.file_path].content[top.start : top.end]
                lines.append(
                    f"- **returned**: `{top.file_path.split('/')[-1][:60]}` "
                    f"[{top.start}:{top.end}] score={top.score:.3f} — {excerpt(text, 200)}"
                )
            else:
                lines.append("- **returned**: nothing (abstained)")
            lines.append(f"- **recall@{args.k}**: {score.recall:.3f}")
            lines.append(f"- **diagnosis**: {DIAGNOSIS[cause]}")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for cause in CATEGORY_ORDER:
        if failures.get(cause):
            print(f"{cause:18s} {len(failures[cause]):3d}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
