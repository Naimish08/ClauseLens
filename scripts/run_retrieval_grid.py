#!/usr/bin/env python3
"""Run the retrieval comparison grid with chunking held fixed.

    python scripts/run_retrieval_grid.py --config configs/grid_retrieval.yaml

Each variant is an explicit retrieval spec, so the table says exactly what ran.
Reports quality at several k plus latency, because a reranker that adds two
seconds must justify itself.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import EvalResult, evaluate, percentile  # noqa: E402
from src.config import parse_config  # noqa: E402
from src.data import ARTIFACTS_DIR, load_benchmark_split  # noqa: E402

METRICS = ("recall", "precision", "f1", "span_hit", "chunk_hit", "doc_hit", "mrr")


def variant_config(grid: dict[str, Any], variant: dict[str, Any]):
    raw = {
        "name": variant["name"],
        "seed": grid.get("seed", 13),
        "data": {
            "benchmark": grid["data"]["benchmark"],
            "mini_limit": grid["data"]["mini_limit"],
            "read_mode": grid["data"]["read_mode"],
            "split": grid.get("split", "train"),
        },
        "chunking": grid["chunking"],
        "retrieval": variant["retrieval"],
        "rerank": grid.get("rerank", {"name": "none"}),
        "abstain": grid.get("abstain", {"name": "none"}),
        "query": grid.get("query", {"strip_doc_hint": False}),
        "metrics": {"mode": grid.get("mode", "paper"), "ks": grid["ks"]},
    }
    return parse_config(raw)


def row_from(result: EvalResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": result.config.name,
        "retriever": result.config.retrieval.name,
        "index_ms": round(result.index_ms, 1),
        "latency_p50_ms": round(percentile(result.latencies_ms, 50), 2),
        "latency_p95_ms": round(percentile(result.latencies_ms, 95), 2),
    }
    for k in result.ks:
        aggregate = result.aggregates[k]
        for metric in METRICS:
            row[f"{metric}@{k}"] = round(getattr(aggregate, metric), 4)
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]], k: int) -> str:
    lines = [
        f"| retriever | recall@{k} | precision@{k} | mrr@{k} | span_hit@{k} "
        f"| chunk_hit@{k} | latency p50 (ms) | latency p95 (ms) | index (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row[f'recall@{k}']:.3f} | {row[f'precision@{k}']:.3f} "
            f"| {row[f'mrr@{k}']:.3f} | {row[f'span_hit@{k}']:.3f} "
            f"| {row[f'chunk_hit@{k}']:.3f} | {row['latency_p50_ms']:.2f} "
            f"| {row['latency_p95_ms']:.2f} | {row['index_ms']:.0f} |"
        )
    return "\n".join(lines)


def plot(rows: list[dict[str, Any]], k: int, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return False

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for row in rows:
        axis.scatter(row["latency_p50_ms"], row[f"recall@{k}"], s=60)
        axis.annotate(
            row["variant"],
            (row["latency_p50_ms"], row[f"recall@{k}"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    axis.set_xlabel("median latency per query (ms)")
    axis.set_ylabel(f"character recall@{k}")
    axis.set_title(f"Quality vs latency ({len(rows)} retrieval configurations, train)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/grid_retrieval.yaml"))
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "grids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    grid = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    queries, documents = load_benchmark_split(
        benchmark=grid["data"]["benchmark"],
        mini_limit=grid["data"]["mini_limit"],
        read_mode=grid["data"]["read_mode"],
        split=grid.get("split", "train"),
    )
    if args.limit:
        queries = queries[: args.limit]

    report_k = 10
    rows: list[dict[str, Any]] = []

    for variant in grid["variants"]:
        config = variant_config(grid, variant)
        try:
            result = evaluate(
                config,
                queries,
                documents,
                ks=tuple(grid["ks"]),
                mode=grid.get("mode", "paper"),
            )
        except ImportError as exc:
            print(
                f"{config.name}: missing dependency ({exc}); "
                "install the ML extra with `uv pip install -e '.[ml]'`",
                file=sys.stderr,
            )
            return 2
        row = row_from(result)
        rows.append(row)
        print(
            f"{row['variant']:20s} R@{report_k}={row[f'recall@{report_k}']:.3f} "
            f"P@{report_k}={row[f'precision@{report_k}']:.3f} "
            f"mrr={row[f'mrr@{report_k}']:.3f} "
            f"p50={row['latency_p50_ms']:7.2f} ms  index={row['index_ms']:.0f} ms"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "retrieval_grid.csv")
    table = render_markdown(rows, report_k)
    (args.out_dir / "retrieval_grid.md").write_text(
        f"# Retrieval grid (`{grid.get('mode', 'paper')}`, split `{grid.get('split')}`)\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    plotted = False if args.no_plot else plot(rows, report_k, args.out_dir / "retrieval_grid.png")

    print()
    print(table)
    print(f"\nwrote {args.out_dir / 'retrieval_grid.csv'}")
    if plotted:
        print(f"wrote {args.out_dir / 'retrieval_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
