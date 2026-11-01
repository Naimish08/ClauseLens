#!/usr/bin/env python3
"""Evaluate a pipeline config and write reproducible run artefacts.

    python -m eval.run_eval --config configs/baseline.yaml
    python -m eval.run_eval --config configs/baseline.yaml --split test --i-know-this-is-final

Outputs land in ``artifacts/runs/{run_id}/``:

    config.yaml      a copy of the exact config used
    per_query.jsonl  one line per query, with every metric at every k
    aggregate.json   macro-averaged metrics, per k and per category
    table.md         the markdown table for the report
    table.csv        the same numbers for plotting
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import MetricMode, PipelineConfig, load_config
from src.data import ARTIFACTS_DIR, REPO_ROOT, Document, Query, load_benchmark_split
from src.metrics import (
    Aggregate,
    ContainmentReport,
    QueryScore,
    aggregate,
    aggregate_by_category,
    score_query,
    span_containment,
)
from src.pipeline import Pipeline

TEST_RUN_LOG = REPO_ROOT / "logs" / "test_runs.log"
AGGREGATE_COLUMNS = (
    "recall",
    "precision",
    "f1",
    "doc_hit",
    "span_hit",
    "chunk_hit",
    "mrr",
    "abstention_rate",
)


@dataclass
class EvalResult:
    run_id: str
    config: PipelineConfig
    split: str
    ks: tuple[int, ...]
    mode: MetricMode
    scores: dict[int, list[QueryScore]]
    aggregates: dict[int, Aggregate]
    per_category: dict[str, Aggregate]
    containment: ContainmentReport
    index_ms: float
    latencies_ms: list[float]
    index_stats: dict[str, int]
    run_dir: Path

    @property
    def headline(self) -> Aggregate:
        return self.aggregates[self.config.top_k]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.as_dict(),
            "config_hash": self.config.hash(),
            "split": self.split,
            "mode": self.mode,
            "ks": list(self.ks),
            "index_stats": self.index_stats,
            "span_containment": self.containment.as_dict(),
            "index_ms": self.index_ms,
            "latency_ms": {
                "median": percentile(self.latencies_ms, 50),
                "p95": percentile(self.latencies_ms, 95),
                "n": len(self.latencies_ms),
            },
            "aggregates": {str(k): agg.as_dict() for k, agg in self.aggregates.items()},
            "per_category": {cat: agg.as_dict() for cat, agg in self.per_category.items()},
        }


def percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else 0.0


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def evaluate(
    config: PipelineConfig,
    queries: list[Query],
    documents: dict[str, Document],
    *,
    mode: MetricMode | None = None,
    ks: tuple[int, ...] | None = None,
) -> EvalResult:
    """Run one config over a query list and score every query."""
    metric_mode: MetricMode = mode or config.metrics.mode
    cutoffs = tuple(sorted(set(ks or config.metrics.ks) | {config.top_k}))

    import time as _time

    pipeline = Pipeline.build(config)
    _index_started = _time.perf_counter()
    index_stats = pipeline.index(documents.values())
    index_ms = (_time.perf_counter() - _index_started) * 1000.0
    containment = span_containment(queries, pipeline.chunks)

    scores: dict[int, list[QueryScore]] = {k: [] for k in cutoffs}
    latencies: list[float] = []

    for query in queries:
        result = pipeline.search(query.text, k=max(cutoffs))
        latencies.append(sum(result.timings_ms.values()))
        for k in cutoffs:
            scores[k].append(score_query(query, result.spans, k=k, mode=metric_mode))

    aggregates = {k: aggregate(scores[k]) for k in cutoffs}

    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    config_name = config.name.replace(" ", "-")
    run_id = f"{config_name}-{config.data.split}-{config.hash()}-{created}"

    return EvalResult(
        run_id=run_id,
        config=config,
        split=config.data.split,
        ks=cutoffs,
        mode=metric_mode,
        scores=scores,
        aggregates=aggregates,
        per_category=aggregate_by_category(scores[config.top_k]),
        containment=containment,
        index_ms=index_ms,
        latencies_ms=latencies,
        index_stats=index_stats.as_dict(),
        run_dir=ARTIFACTS_DIR / "runs" / run_id,
    )


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #
def write_artefacts(result: EvalResult, config_path: Path) -> None:
    result.run_dir.mkdir(parents=True, exist_ok=True)

    (result.run_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with (result.run_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for query_index in range(len(result.scores[result.ks[0]])):
            record = {
                "query": result.scores[result.ks[0]][query_index].query,
                "category": result.scores[result.ks[0]][query_index].category,
                "split": result.scores[result.ks[0]][query_index].split,
                "ks": {
                    str(k): result.scores[k][query_index].as_dict() for k in result.ks
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (result.run_dir / "aggregate.json").write_text(
        json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (result.run_dir / "table.md").write_text(render_markdown(result), encoding="utf-8")
    write_csv(result, result.run_dir / "table.csv")


def render_markdown(result: EvalResult) -> str:
    header = "| k | " + " | ".join(AGGREGATE_COLUMNS) + " |"
    divider = "|---|" + "---|" * len(AGGREGATE_COLUMNS)
    rows = [header, divider]
    for k in result.ks:
        agg = result.aggregates[k]
        cells = " | ".join(f"{getattr(agg, column):.3f}" for column in AGGREGATE_COLUMNS)
        rows.append(f"| {k} | {cells} |")

    lines = [
        f"# {result.config.name}",
        "",
        f"- split: `{result.split}` ({result.aggregates[result.ks[0]].n} queries)",
        f"- metric mode: `{result.mode}`",
        f"- index: {result.index_stats['documents']} docs, "
        f"{result.index_stats['chunks']} chunks, {result.index_stats['characters']} chars",
        f"- span containment: {result.containment.mean:.3f} "
        f"({result.containment.full}/{result.containment.total} GT spans fully inside a chunk)",
        f"- latency: median {percentile(result.latencies_ms, 50):.1f} ms, "
        f"p95 {percentile(result.latencies_ms, 95):.1f} ms",
        f"- index build: {result.index_ms:.0f} ms",
        "",
        *rows,
        "",
        f"_run_id: `{result.run_id}`_",
        "",
    ]
    return "\n".join(lines)


def write_csv(result: EvalResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", *AGGREGATE_COLUMNS, "n"])
        for k in result.ks:
            agg = result.aggregates[k]
            writer.writerow([k, *(f"{getattr(agg, c):.6f}" for c in AGGREGATE_COLUMNS), agg.n])


def log_test_run(result: EvalResult, reason: str) -> None:
    TEST_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": result.run_id,
        "config": result.config.name,
        "config_hash": result.config.hash(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "reason": reason,
    }
    with TEST_RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default=None)
    parser.add_argument("--mode", choices=("paper", "union"), default=None)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N queries")
    parser.add_argument(
        "--i-know-this-is-final",
        action="store_true",
        help="required to evaluate the test split; the run is recorded in logs/test_runs.log",
    )
    parser.add_argument("--reason", default="final evaluation")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.split:
        config = _with_split(config, args.split)

    if config.data.split == "test" and not args.i_know_this_is_final:
        print(
            "refusing to evaluate the test split without --i-know-this-is-final\n"
            "test is for final reporting only (docs/08-evaluation-protocol.md)",
            file=sys.stderr,
        )
        return 2

    queries, documents = load_benchmark_split(
        benchmark=config.data.benchmark,
        mini_limit=config.data.mini_limit,
        read_mode=config.data.read_mode,
        split=config.data.split,
    )
    if args.limit:
        queries = queries[: args.limit]

    result = evaluate(
        config,
        queries,
        documents,
        mode=args.mode,
        ks=tuple(config.metrics.ks),
    )
    write_artefacts(result, args.config)

    if config.data.split == "test":
        log_test_run(result, args.reason)

    if not args.quiet:
        print(render_markdown(result))
        print(f"artefacts: {result.run_dir}")

    return 0


def _with_split(config: PipelineConfig, split: str) -> PipelineConfig:
    import dataclasses

    return dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, split=split),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    raise SystemExit(main())
