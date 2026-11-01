"""Improvement loop: evaluate, diagnose, propose, accept, log.

    python -m loop.run_loop --config configs/baseline.yaml --iterations 6

Train only: the loop refuses a config whose split is ``test``. Every iteration,
accepted or rejected, is appended to ``logs/experiments.jsonl`` (schema in
docs/10-experiment-log.md). Accepted candidates are written to
``configs/generated/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import percentile  # noqa: E402
from loop.accept import evaluate_gates  # noqa: E402
from loop.diagnose import Diagnosis, diagnose  # noqa: E402
from loop.propose import Proposal, behaviour_signature, propose  # noqa: E402
from src.config import PipelineConfig, load_config  # noqa: E402
from src.data import (  # noqa: E402
    REPO_ROOT,
    fold_of,
    load_benchmark_split,
    load_mini_cuad,
)
from src.metrics import QueryScore, aggregate, score_query  # noqa: E402
from src.negatives import NegativeQuery, load_negatives  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

PRIMARY_K = 10
NEGATIVES_PATH = REPO_ROOT / "data" / "unanswerable.jsonl"


@dataclass
class Outcome:
    scores: list[QueryScore]
    latencies_ms: list[float]
    index_ms: float
    false_abstention_rate: float
    correct_abstention_rate: float

    @property
    def recall(self) -> float:
        return aggregate(self.scores).recall

    @property
    def precision(self) -> float:
        return aggregate(self.scores).precision

    @property
    def latency_p95(self) -> float:
        return percentile(self.latencies_ms, 95)


def run_config(
    config: PipelineConfig,
    queries,
    documents,
    negatives: list[NegativeQuery],
) -> Outcome:
    """One pass over train answerable queries and train negatives."""
    import time

    pipeline = Pipeline.build(config)
    started = time.perf_counter()
    pipeline.index(documents.values())
    index_ms = (time.perf_counter() - started) * 1000.0

    depth = max(config.metrics.ks)
    scores: list[QueryScore] = []
    latencies: list[float] = []
    for query in queries:
        result = pipeline.search(query.text, k=depth)
        scores.append(score_query(query, result.spans, k=PRIMARY_K))
        latencies.append(sum(result.timings_ms.values()))

    negative_abstained = [
        len(pipeline.search(negative.query, k=depth).spans) == 0 for negative in negatives
    ]

    return Outcome(
        scores=scores,
        latencies_ms=latencies,
        index_ms=index_ms,
        false_abstention_rate=aggregate(scores).false_abstention_rate,
        correct_abstention_rate=(
            float(np.mean(negative_abstained)) if negative_abstained else 0.0
        ),
    )


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def write_config(config: PipelineConfig, path: Path, *, name: str | None = None) -> None:
    """Persist a config, optionally renaming it so runs are distinguishable."""
    payload = config.as_dict()
    if name is not None:
        payload["name"] = name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def log_record(
    log_path: Path,
    *,
    iteration: int,
    accepted: bool,
    proposal: str,
    trigger: str,
    reason: str,
    diff: dict[str, Any],
    config_path: str | None,
    outcome: Outcome,
    diagnosis: Diagnosis,
    gates: dict[str, Any] | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "iteration": iteration,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "accepted": accepted,
        "invalid": False,
        "proposal": {
            "proposer": proposal,
            "trigger": trigger,
            "reason": reason,
            "diff": diff,
            "config_path": config_path,
        },
        "evaluation": {
            "split": "train",
            "n_queries": len(outcome.scores),
            "primary_metric": f"recall@{PRIMARY_K}",
            "primary": round(outcome.recall, 4),
            "primary_delta": None if gates is None else gates["delta"],
            "primary_ci95": None if gates is None else gates["ci95"],
            "guardrails": {
                "precision@10": round(outcome.precision, 4),
                "false_abstention_rate": round(outcome.false_abstention_rate, 4),
                "latency_ms_p95": round(outcome.latency_p95, 2),
            },
            "secondary": {
                "mrr@10": round(aggregate(outcome.scores).mrr, 4),
                "span_hit@10": round(aggregate(outcome.scores).span_hit, 4),
                "chunk_hit@10": round(aggregate(outcome.scores).chunk_hit, 4),
            },
            "index_ms": round(outcome.index_ms, 1),
            "correct_abstention_rate": round(outcome.correct_abstention_rate, 4),
        },
        "diagnosis": diagnosis.as_dict(),
        "decision": gates or {"stage": "baseline"},
        "git_commit": git_sha(),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-path", type=Path, default=REPO_ROOT / "logs" / "experiments.jsonl")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs" / "generated")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config.data.split == "test":
        print("refusing to run the loop against the test split", file=sys.stderr)
        return 2
    if config.data.split != "train":
        config = dataclasses.replace(
            config, data=dataclasses.replace(config.data, split="train")
        )

    queries, _ = load_benchmark_split(
        benchmark=config.data.benchmark,
        mini_limit=config.data.mini_limit,
        read_mode=config.data.read_mode,
        split="train",
    )
    if args.limit:
        queries = queries[: args.limit]
    _, documents = load_mini_cuad(mode=config.data.read_mode)
    negatives = [n for n in load_negatives(NEGATIVES_PATH) if n.split == "train"]

    folds = [fold_of(query.text) for query in queries]

    print(
        f"loop over {len(queries)} train queries, {len(negatives)} train negatives, "
        f"primary metric recall@{PRIMARY_K}"
    )

    current = run_config(config, queries, documents, negatives)
    baseline_mean_len = float(
        np.mean([score.retrieved_len for score in current.scores if score.answerable])
    )
    diagnosis = diagnose(current.scores)
    log_record(
        args.log_path,
        iteration=0,
        accepted=True,
        proposal="baseline",
        trigger="n/a",
        reason="initial reference pipeline",
        diff={},
        config_path=str(args.config),
        outcome=current,
        diagnosis=diagnosis,
        gates=None,
    )
    print(
        f"[0] baseline {config.name}: recall@{PRIMARY_K}={current.recall:.3f} "
        f"precision={current.precision:.3f} p95={current.latency_p95:.1f} ms "
        f"dominant={diagnosis.dominant}"
    )

    tried_rules: set[str] = set()
    tried_signatures: set[str] = {behaviour_signature(config)}
    accepted_count = 0
    history: list[tuple[int, str, str, float, float, bool]] = []

    for iteration in range(1, args.iterations + 1):
        diagnosis = diagnose(current.scores)
        proposal: Proposal | None = propose(
            config, diagnosis.dominant, tried_rules, tried_signatures
        )
        if proposal is None:
            print(f"[{iteration}] no untried proposal for {diagnosis.dominant}; stopping")
            break

        candidate = run_config(proposal.config, queries, documents, negatives)
        gates = evaluate_gates(
            current.scores,
            candidate.scores,
            folds,
            seed=config.seed,
            latency_current_p95=current.latency_p95,
            latency_candidate_p95=candidate.latency_p95,
            baseline_mean_retrieved_len=baseline_mean_len,
        )
        candidate_diagnosis = diagnose(candidate.scores)

        accepted = gates.accepted
        config_path: str | None = None
        if accepted:
            accepted_name = f"iter{iteration:02d}-{proposal.rule}"
            config_path = str(args.config_dir / f"iter_{iteration:02d}.yaml")
            write_config(proposal.config, Path(config_path), name=accepted_name)
            current = candidate
            config = dataclasses.replace(proposal.config, name=accepted_name)
            accepted_count += 1
            tried_signatures.add(behaviour_signature(proposal.config))
            # The baseline changed, so earlier rejections were measured against a
            # different system. Re-open them; exact repeats are still blocked by
            # tried_signatures.
            tried_rules.clear()
        else:
            tried_rules.add(proposal.rule)
            tried_signatures.add(behaviour_signature(proposal.config))

        log_record(
            args.log_path,
            iteration=iteration,
            accepted=accepted,
            proposal=f"rule:{proposal.rule}",
            trigger=diagnosis.dominant or "n/a",
            reason=proposal.reason,
            diff=proposal.diff,
            config_path=config_path,
            outcome=candidate,
            diagnosis=candidate_diagnosis,
            gates=gates.as_dict(),
        )
        history.append(
            (iteration, proposal.rule, gates.notes, gates.delta, candidate.recall, accepted)
        )
        print(
            f"[{iteration:2d}] {proposal.rule:16s} {gates.notes:60s} "
            f"delta={gates.delta:+.3f} recall={candidate.recall:.3f}"
        )

    print(
        f"\n{accepted_count} accepted of {len(history)} iterations; "
        f"final config recall@{PRIMARY_K}={current.recall:.3f} "
        f"precision={current.precision:.3f}"
    )
    print(f"log: {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
