#!/usr/bin/env python3
"""Run the chunking grid and write the comparison table plus the centrepiece plot.

    python scripts/run_grid.py --config configs/grid_chunk.yaml

Every variant is scored by the same metric code and the same pipeline as the
baseline; only the chunker parameters change. Alongside retrieval metrics the grid
reports **span containment** — the fraction of ground-truth answers that sit fully
inside a single chunk — which separates chunking failures from ranking failures
without running a retriever at all.
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


def chunk_params(chunker: str, size: int, overlap_chars: int) -> dict[str, Any]:
    """Map the grid's common (size, overlap) axes onto each chunker's parameters.

    Sizes are characters everywhere (ADR-013). For the clause chunker ``size`` is
    the maximum section size; ``min_size`` keeps sections from fragmenting.
    """
    if chunker == "fixed":
        return {"size": size, "overlap": overlap_chars, "snap": True, "snap_tolerance": 80}
    if chunker in {"recursive", "sentence"}:
        return {"size": size, "overlap": overlap_chars}
    if chunker == "clause":
        return {"max_size": size, "min_size": min(200, size), "overlap": overlap_chars}
    raise ValueError(f"unknown chunker in grid: {chunker!r}")


def variant_config(grid: dict[str, Any], chunker: str, size: int, overlap_fraction: float):
    overlap_chars = int(size * overlap_fraction)
    name = f"{chunker}-{size}-ov{int(overlap_fraction * 100):02d}"
    raw = {
        "name": name,
        "seed": grid.get("seed", 13),
        "data": {
            "benchmark": grid["data"]["benchmark"],
            "mini_limit": grid["data"]["mini_limit"],
            "read_mode": grid["data"]["read_mode"],
            "split": grid.get("split", "train"),
        },
        "chunking": {
            "name": chunker,
            "params": chunk_params(chunker, size, overlap_chars),
        },
        "retrieval": {
            "name": grid["retrieval"]["name"],
            "params": grid["retrieval"]["params"],
            "top_k": grid["retrieval"].get("top_k", 10),
        },
        "rerank": {"name": "none"},
        "abstain": {"name": "none"},
        "query": {"strip_doc_hint": grid.get("strip_doc_hint", False)},
        "metrics": {"mode": grid.get("mode", "paper"), "ks": grid["ks"]},
    }
    return parse_config(raw)


def row_from(
    result: EvalResult, chunker: str, size: int, overlap_fraction: float
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": result.config.name,
        "chunker": chunker,
        "size": size,
        "overlap_pct": int(overlap_fraction * 100),
        "n_chunks": result.index_stats["chunks"],
        "mean_chunk_chars": round(
            result.index_stats["characters"] / max(1, result.index_stats["chunks"]), 1
        ),
        "containment": round(result.containment.mean, 4),
        "latency_ms_p50": round(percentile(result.latencies_ms, 50), 3),
    }
    for k in result.ks:
        aggregate = result.aggregates[k]
        for metric in ("recall", "precision", "f1", "span_hit", "chunk_hit", "doc_hit", "mrr"):
            row[f"{metric}@{k}"] = round(getattr(aggregate, metric), 4)
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]], k: int) -> str:
    header = (
        "| chunker | size | overlap | chunks | mean len | containment "
        f"| recall@{k} | precision@{k} | span_hit@{k} | chunk_hit@{k} |"
    )
    divider = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for row in rows:
        lines.append(
            f"| {row['chunker']} | {row['size']} | {row['overlap_pct']}% | {row['n_chunks']} "
            f"| {row['mean_chunk_chars']:.0f} | {row['containment']:.3f} "
            f"| {row[f'recall@{k}']:.3f} | {row[f'precision@{k}']:.3f} "
            f"| {row[f'span_hit@{k}']:.3f} | {row[f'chunk_hit@{k}']:.3f} |"
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

    chunkers = sorted({row["chunker"] for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for chunker in chunkers:
        series = sorted(
            (row for row in rows if row["chunker"] == chunker),
            key=lambda row: row["mean_chunk_chars"],
        )
        x = [row["mean_chunk_chars"] for row in series]
        axes[0].plot(x, [row[f"recall@{k}"] for row in series], marker="o", label=chunker)
        axes[1].plot(x, [row[f"precision@{k}"] for row in series], marker="o", label=chunker)

    axes[0].set_title(f"Character recall@{k} vs chunk size")
    axes[1].set_title(f"Character precision@{k} vs chunk size")
    for axis in axes:
        axis.set_xlabel("mean chunk length (characters, log scale)")
        axis.set_xscale("log")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("recall")
    axes[1].set_ylabel("precision")
    axes[0].legend(title="chunker", fontsize=8)
    figure.suptitle(
        "Chunking trade-off (BM25, CUAD-mini train, median ground-truth answer 317 chars)"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/grid_chunk.yaml"))
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

    report_k = grid["retrieval"].get("top_k", 10)
    rows: list[dict[str, Any]] = []

    for chunker in grid["chunkers"]:
        for size in grid["sizes"]:
            for overlap_fraction in grid["overlaps"]:
                config = variant_config(grid, chunker, size, overlap_fraction)
                result = evaluate(
                    config,
                    queries,
                    documents,
                    ks=tuple(grid["ks"]),
                    mode=grid.get("mode", "paper"),
                )
                row = row_from(result, chunker, size, overlap_fraction)
                rows.append(row)
                print(
                    f"{row['variant']:22s} chunks={row['n_chunks']:5d} "
                    f"mean={row['mean_chunk_chars']:6.0f} cont={row['containment']:.3f} "
                    f"R@{report_k}={row[f'recall@{report_k}']:.3f} "
                    f"P@{report_k}={row[f'precision@{report_k}']:.3f}"
                )

    if not rows:
        print("no variants configured", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "chunk_grid.csv")
    table = render_markdown(rows, report_k)
    (args.out_dir / "chunk_grid.md").write_text(
        f"# Chunking grid (`{grid.get('mode', 'paper')}` metric, split `{grid.get('split')}`)\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    plotted = False if args.no_plot else plot(rows, report_k, args.out_dir / "chunk_grid.png")

    print()
    print(table)
    print(f"\nwrote {args.out_dir / 'chunk_grid.csv'}")
    print(f"wrote {args.out_dir / 'chunk_grid.md'}")
    if plotted:
        print(f"wrote {args.out_dir / 'chunk_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
