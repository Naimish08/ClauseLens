#!/usr/bin/env python3
"""Paired comparison of two configs with bootstrap intervals.

    python scripts/compare_configs.py --a configs/baseline.yaml --b configs/final.yaml

Both configs are evaluated on the same queries, so every metric is paired by
query. For each metric the script reports both means with 95% percentile
intervals and the paired difference with its interval. A difference is only
reported as significant when the interval excludes zero.

This is the tool behind every "final beats baseline by ..." claim in the report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import evaluate  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import ARTIFACTS_DIR, load_benchmark_split  # noqa: E402
from src.metrics import QueryScore, bootstrap_ci, paired_bootstrap  # noqa: E402

METRICS: tuple[str, ...] = (
    "recall",
    "precision",
    "f1",
    "mrr",
    "span_hit",
    "chunk_hit",
    "doc_hit",
)


def vector(scores: list[QueryScore], metric: str) -> list[float]:
    return [float(getattr(score, metric)) for score in scores if score.answerable]


def compare(
    a_scores: list[QueryScore],
    b_scores: list[QueryScore],
    *,
    k: int,
    seed: int,
    n_resamples: int,
) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        a_values = vector(a_scores, metric)
        b_values = vector(b_scores, metric)
        a_mean, a_lo, a_hi = bootstrap_ci(a_values, n_resamples=n_resamples, seed=seed)
        b_mean, b_lo, b_hi = bootstrap_ci(b_values, n_resamples=n_resamples, seed=seed)
        paired = paired_bootstrap(b_values, a_values, n_resamples=n_resamples, seed=seed)
        rows.append(
            {
                "metric": f"{metric}@{k}",
                "a": a_mean,
                "a_ci": (a_lo, a_hi),
                "b": b_mean,
                "b_ci": (b_lo, b_hi),
                "delta": paired.delta,
                "delta_ci": (paired.lo, paired.hi),
                "significant": paired.significant,
            }
        )
    return rows


def render(rows: list[dict[str, Any]], name_a: str, name_b: str) -> str:
    lines = [
        f"| metric | {name_a} | {name_b} | delta (B − A) | 95% CI on delta | significant |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} "
            f"| {row['a']:.3f} [{row['a_ci'][0]:.3f}, {row['a_ci'][1]:.3f}] "
            f"| {row['b']:.3f} [{row['b_ci'][0]:.3f}, {row['b_ci'][1]:.3f}] "
            f"| {row['delta']:+.4f} "
            f"| [{row['delta_ci'][0]:+.4f}, {row['delta_ci'][1]:+.4f}] "
            f"| {'yes' if row['significant'] else 'no'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--a", type=Path, required=True, help="reference config (e.g. baseline)")
    parser.add_argument("--b", type=Path, required=True, help="candidate config (e.g. final)")
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "comparison.md")
    args = parser.parse_args(argv)

    config_a = load_config(args.a)
    config_b = load_config(args.b)
    split = args.split or config_a.data.split

    queries, documents = load_benchmark_split(
        benchmark=config_a.data.benchmark,
        mini_limit=config_a.data.mini_limit,
        read_mode=config_a.data.read_mode,
        split=split,
    )
    if args.limit:
        queries = queries[: args.limit]

    ks = tuple(sorted({args.k, config_a.top_k, config_b.top_k}))
    result_a = evaluate(config_a, queries, documents, ks=ks)
    result_b = evaluate(config_b, queries, documents, ks=ks)

    rows = compare(
        result_a.scores[args.k],
        result_b.scores[args.k],
        k=args.k,
        seed=args.seed,
        n_resamples=args.resamples,
    )

    table = render(rows, config_a.name, config_b.name)
    header = (
        f"# Paired comparison — `{config_a.name}` vs `{config_b.name}`\n\n"
        f"split: `{split}`, {len(queries)} queries, paired bootstrap "
        f"({args.resamples:,} resamples)\n\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + table + "\n", encoding="utf-8")

    print(table)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
