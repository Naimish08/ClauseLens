#!/usr/bin/env python3
"""Compare abstention policies as trade-off curves.

    python -m eval.run_abstention --config configs/baseline.yaml

Evaluated on **train only** by default: answerable train queries (false abstention)
against the frozen unanswerable set's train split (correct abstention), broken down
by negative type.

``category`` and ``classifier`` policies are fitted, so their confidences are
out-of-fold (5 folds derived from the query hash). ``threshold`` and ``margin``
have no fitted parameters, so their confidences come from a single pass.

The operating point is chosen by a rule fixed in advance: the lowest
false-abstention rate among thresholds where correct abstention is at least
``--target`` (default 0.80), falling back to the best abstention F1 if that is
unreachable.

For final reporting, ``--split test`` applies the threshold already stored in the
config to the test split, with no fitting or threshold selection on test:

    python -m eval.run_abstention --config configs/final.yaml \\
        --split test --i-know-this-is-final
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import log_test_event  # noqa: E402
from src.abstain import FEATURE_NAMES, QueryFeatures  # noqa: E402
from src.config import PipelineConfig, load_config  # noqa: E402
from src.data import (  # noqa: E402
    ARTIFACTS_DIR,
    Document,
    Query,
    fold_of,
    load_benchmark_split,
)
from src.negatives import NegativeQuery, load_negatives  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

NEGATIVES_PATH = Path(__file__).resolve().parents[1] / "data" / "unanswerable.jsonl"
POLICIES = ("threshold", "margin", "category", "classifier")
MIN_CATEGORY_SUPPORT = 8


@dataclass
class Sample:
    features: QueryFeatures
    answerable: bool
    negative_type: int | None
    fold: int


@dataclass
class OperatingPoint:
    policy: str
    threshold: float
    false_abstention_rate: float
    correct_abstention_rate: float
    abstention_precision: float
    abstention_f1: float
    per_type: dict[str, float]


# --------------------------------------------------------------------------- #
# Feature collection
# --------------------------------------------------------------------------- #
def collect_samples(
    config: PipelineConfig,
    queries: list[Query],
    negatives: list[NegativeQuery],
    documents: dict[str, Document] | None = None,
) -> list[Sample]:
    pipeline = Pipeline.build(config)
    if documents is None:
        _, documents = load_benchmark_split(
            benchmark=config.data.benchmark,
            mini_limit=config.data.mini_limit,
            read_mode=config.data.read_mode,
            split="all",
        )
    pipeline.index(documents.values())

    samples = [
        Sample(
            features=QueryFeatures.from_retrieval(
                query.text, pipeline.search(query.text, k=config.top_k).spans
            ),
            answerable=True,
            negative_type=None,
            fold=fold_of(query.text),
        )
        for query in queries
    ]
    samples.extend(
        Sample(
            features=QueryFeatures.from_retrieval(
                negative.query, pipeline.search(negative.query, k=config.top_k).spans
            ),
            answerable=False,
            negative_type=negative.type,
            fold=fold_of(negative.query),
        )
        for negative in negatives
    )
    return samples


# --------------------------------------------------------------------------- #
# Fitted policies (out-of-fold confidences)
# --------------------------------------------------------------------------- #
def category_confidences(samples: list[Sample], n_folds: int) -> list[float]:
    """Normalise top1 by the clause-type scale learned on the other folds."""
    confidences = [0.0] * len(samples)
    for fold in range(n_folds):
        train = [s for s in samples if s.fold != fold and s.answerable]
        global_scale = float(np.quantile([s.features.top1 for s in train], 0.9)) if train else 1.0
        by_category: dict[str, list[float]] = {}
        for sample in train:
            by_category.setdefault(sample.features.category, []).append(sample.features.top1)
        scales = {
            category: float(np.quantile(values, 0.9))
            for category, values in by_category.items()
            if len(values) >= MIN_CATEGORY_SUPPORT
        }
        for index, sample in enumerate(samples):
            if sample.fold != fold:
                continue
            scale = scales.get(sample.features.category, global_scale) or 1.0
            confidences[index] = sample.features.top1 / scale
    return confidences


def classifier_confidences(samples: list[Sample], n_folds: int, seed: int) -> list[float]:
    """Out-of-fold probability that the query is answerable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    confidences = np.zeros(len(samples), dtype=float)
    x_all = np.vstack([sample.features.vector() for sample in samples])
    y_all = np.array([0 if sample.answerable else 1 for sample in samples])
    folds = np.array([sample.fold for sample in samples])

    for fold in range(n_folds):
        test = folds == fold
        train = ~test
        if test.sum() == 0:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
        )
        model.fit(x_all[train], y_all[train])
        probability_unanswerable = model.predict_proba(x_all[test])[:, 1]
        confidences[np.where(test)[0]] = 1.0 - probability_unanswerable
    return confidences.tolist()


def confidences_for(policy: str, samples: list[Sample], n_folds: int, seed: int) -> list[float]:
    if policy == "threshold":
        return [sample.features.top1 for sample in samples]
    if policy == "margin":
        return [sample.features.gap for sample in samples]
    if policy == "category":
        return category_confidences(samples, n_folds)
    if policy == "classifier":
        return classifier_confidences(samples, n_folds, seed)
    raise ValueError(f"unknown policy {policy!r}")


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def harmonic(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def evaluate_threshold(
    confidences: list[float], samples: list[Sample], threshold: float
) -> OperatingPoint:
    answerable = [c for c, s in zip(confidences, samples, strict=True) if s.answerable]
    negatives = [
        (c, s.negative_type) for c, s in zip(confidences, samples, strict=True) if not s.answerable
    ]
    false_abstentions = sum(1 for c in answerable if c < threshold)
    correct = [(c, t) for c, t in negatives if c < threshold]

    far = false_abstentions / len(answerable) if answerable else 0.0
    car = len(correct) / len(negatives) if negatives else 0.0
    total_abstentions = false_abstentions + len(correct)
    precision = len(correct) / total_abstentions if total_abstentions else 0.0

    per_type: dict[str, list[bool]] = {}
    for c, type_id in negatives:
        per_type.setdefault(str(type_id), []).append(c < threshold)

    return OperatingPoint(
        policy="",
        threshold=threshold,
        false_abstention_rate=far,
        correct_abstention_rate=car,
        abstention_precision=precision,
        abstention_f1=harmonic(1.0 - far, car),
        per_type={key: float(np.mean(values)) for key, values in sorted(per_type.items())},
    )


def sweep(
    policy: str, confidences: list[float], samples: list[Sample], n_points: int = 60
) -> list[OperatingPoint]:
    values = sorted(set(confidences))
    if not values:
        return []
    if len(values) > n_points:
        indices = np.linspace(0, len(values) - 1, n_points).round().astype(int)
        values = sorted({values[i] for i in indices})
    points = [evaluate_threshold(confidences, samples, v) for v in values]
    for point in points:
        point.policy = policy
    return points


def choose_operating_point(points: list[OperatingPoint], target: float) -> OperatingPoint:
    candidate = [p for p in points if p.correct_abstention_rate >= target]
    if candidate:
        return min(candidate, key=lambda p: (p.false_abstention_rate, -p.correct_abstention_rate))
    return max(points, key=lambda p: p.abstention_f1)



# --------------------------------------------------------------------------- #
# Nested evaluation (honest operating points)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Decision:
    abstain: bool
    answerable: bool
    negative_type: int | None


def nested_decisions(
    policy: str,
    confidences: list[float],
    samples: list[Sample],
    n_folds: int,
    target: float,
) -> tuple[list[Decision], list[float]]:
    """Choose the threshold on other folds, then apply it to the held-out fold.

    Without this, the operating point is selected on the same queries used to
    report it, which flatters every policy by the same optimistic bias.
    """
    decisions: list[Decision] = []
    thresholds: list[float] = []

    for fold in range(n_folds):
        train_indices = [i for i, sample in enumerate(samples) if sample.fold != fold]
        test_indices = [i for i, sample in enumerate(samples) if sample.fold == fold]
        if not train_indices or not test_indices:
            continue
        train_samples = [samples[i] for i in train_indices]
        train_confidences = [confidences[i] for i in train_indices]
        point = choose_operating_point(
            sweep(policy, train_confidences, train_samples), target
        )
        thresholds.append(point.threshold)
        for index in test_indices:
            decisions.append(
                Decision(
                    abstain=confidences[index] < point.threshold,
                    answerable=samples[index].answerable,
                    negative_type=samples[index].negative_type,
                )
            )
    return decisions, thresholds


def summarise(decisions: list[Decision], threshold: float) -> OperatingPoint:
    answerable = [d for d in decisions if d.answerable]
    negatives = [d for d in decisions if not d.answerable]
    false_abstentions = sum(1 for d in answerable if d.abstain)
    correct = [d for d in negatives if d.abstain]

    far = false_abstentions / len(answerable) if answerable else 0.0
    car = len(correct) / len(negatives) if negatives else 0.0
    total = false_abstentions + len(correct)
    precision = len(correct) / total if total else 0.0

    per_type: dict[str, list[bool]] = {}
    for decision in negatives:
        per_type.setdefault(str(decision.negative_type), []).append(decision.abstain)

    return OperatingPoint(
        policy="",
        threshold=threshold,
        false_abstention_rate=far,
        correct_abstention_rate=car,
        abstention_precision=precision,
        abstention_f1=harmonic(1.0 - far, car),
        per_type={k: float(np.mean(v)) for k, v in sorted(per_type.items())},
    )


def operating_point_ci(
    decisions: list[Decision], *, n_resamples: int = 2000, seed: int = 13
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap intervals for false and correct abstention on the same decisions."""
    from src.metrics import bootstrap_ci

    return {
        "false_abstention_rate": bootstrap_ci(
            [1.0 if d.abstain else 0.0 for d in decisions if d.answerable],
            n_resamples=n_resamples,
            seed=seed,
        ),
        "correct_abstention_rate": bootstrap_ci(
            [1.0 if d.abstain else 0.0 for d in decisions if not d.answerable],
            n_resamples=n_resamples,
            seed=seed,
        ),
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_artefacts(
    curves: dict[str, list[OperatingPoint]],
    operating: dict[str, OperatingPoint],
    out_dir: Path,
    *,
    nested: dict[str, OperatingPoint] | None = None,
    intervals: dict[str, dict[str, tuple[float, float, float]]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "abstention_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "policy",
                "threshold",
                "false_abstention_rate",
                "correct_abstention_rate",
                "abstention_precision",
                "abstention_f1",
            ]
        )
        for policy, points in curves.items():
            for point in points:
                writer.writerow(
                    [
                        policy,
                        f"{point.threshold:.6f}",
                        f"{point.false_abstention_rate:.6f}",
                        f"{point.correct_abstention_rate:.6f}",
                        f"{point.abstention_precision:.6f}",
                        f"{point.abstention_f1:.6f}",
                    ]
                )

    lines = [
        "# Abstention operating points (train)",
        "",
        "Rule: lowest false abstention among thresholds with correct abstention >= target,",
        "otherwise highest abstention F1.",
        "",
        "| policy | threshold | false abst. | correct abst. | abst. precision "
        "| abst. F1 | per type |",
        "|---|---|---|---|---|---|---|",
    ]
    for policy, point in operating.items():
        per_type = ", ".join(f"{k}:{v:.2f}" for k, v in point.per_type.items())
        lines.append(
            f"| {policy} | {point.threshold:.4f} | {point.false_abstention_rate:.3f} "
            f"| {point.correct_abstention_rate:.3f} | {point.abstention_precision:.3f} "
            f"| {point.abstention_f1:.3f} | {per_type} |"
        )
    if nested:
        lines += [
            "",
            "## Nested estimate (threshold chosen on other folds)",
            "",
            "| policy | false abst. | 95% CI | correct abst. | 95% CI | abst. F1 |",
            "|---|---|---|---|---|---|",
        ]
        for policy, point in nested.items():
            far_ci = (intervals or {}).get(policy, {}).get(
                "false_abstention_rate", (0.0, 0.0, 0.0)
            )
            car_ci = (intervals or {}).get(policy, {}).get(
                "correct_abstention_rate", (0.0, 0.0, 0.0)
            )
            lines.append(
                f"| {policy} | {point.false_abstention_rate:.3f} "
                f"| [{far_ci[1]:.3f}, {far_ci[2]:.3f}] "
                f"| {point.correct_abstention_rate:.3f} "
                f"| [{car_ci[1]:.3f}, {car_ci[2]:.3f}] | {point.abstention_f1:.3f} |"
            )

    (out_dir / "abstention_operating_points.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    (out_dir / "abstention_summary.json").write_text(
        json.dumps(
            {
                "operating_points_selected_on_train": {k: vars(v) for k, v in operating.items()},
                "operating_points_nested": {k: vars(v) for k, v in (nested or {}).items()},
                "feature_names": list(FEATURE_NAMES),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6.4, 5.2))
        for policy, points in curves.items():
            axis.plot(
                [1 - p.false_abstention_rate for p in points],
                [p.correct_abstention_rate for p in points],
                label=policy,
            )
        axis.set_xlabel("answerable queries answered (1 - false abstention)")
        axis.set_ylabel("unanswerable queries correctly abstained")
        axis.set_title("Abstention trade-off (CUAD-mini train)")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(out_dir / "abstention_curves.png", dpi=140)
        plt.close(figure)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Test-split reporting (fixed threshold, no fitting)
# --------------------------------------------------------------------------- #
def decisions_at_threshold(samples: list[Sample], threshold: float) -> list[Decision]:
    """Apply a fixed threshold to already-collected samples."""
    return [
        Decision(
            abstain=sample.features.top1 < threshold,
            answerable=sample.answerable,
            negative_type=sample.negative_type,
        )
        for sample in samples
    ]


def evaluate_fixed_threshold(
    config: PipelineConfig,
    queries: list[Query],
    negatives: list[NegativeQuery],
    documents: dict[str, Document],
    threshold: float,
) -> tuple[OperatingPoint, dict[str, tuple[float, float, float]]]:
    """Apply an already-selected threshold to a held-out split.

    No policy is fitted and no operating point is chosen here; the threshold is
    the one stored in the config. This is what makes the test result a report
    rather than a second round of tuning.
    """
    samples = collect_samples(config, queries, negatives, documents)
    decisions = decisions_at_threshold(samples, threshold)
    return summarise(decisions, threshold), operating_point_ci(decisions, seed=config.seed)


def write_test_artefacts(
    point: OperatingPoint,
    intervals: dict[str, tuple[float, float, float]],
    out_dir: Path,
    *,
    split: str,
    n_answerable: int,
    n_unanswerable: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    far = intervals["false_abstention_rate"]
    car = intervals["correct_abstention_rate"]
    rows = [
        ("false_abstention_rate", point.false_abstention_rate, far, n_answerable),
        ("correct_abstention_rate", point.correct_abstention_rate, car, n_unanswerable),
    ]

    with (out_dir / "abstention_test.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "ci_low", "ci_high", "n"])
        for name, value, interval, count in rows:
            writer.writerow(
                [name, f"{value:.6f}", f"{interval[1]:.6f}", f"{interval[2]:.6f}", count]
            )

    payload = {
        "split": split,
        "threshold": point.threshold,
        "false_abstention_rate": point.false_abstention_rate,
        "correct_abstention_rate": point.correct_abstention_rate,
        "abstention_precision": point.abstention_precision,
        "abstention_f1": point.abstention_f1,
        "per_type_correct_abstention": point.per_type,
        "intervals": {
            "false_abstention_rate": list(far),
            "correct_abstention_rate": list(car),
        },
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
    }
    (out_dir / "abstention_test.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# Test-split abstention ({split})",
        "",
        f"Threshold fixed at `{point.threshold:.4f}` from the evaluated config; "
        "no threshold selection on test.",
        "",
        "| direction | rate | 95% CI | n |",
        "|---|---|---|---|",
        f"| false abstention (answerable) | {point.false_abstention_rate:.3f} "
        f"| [{far[1]:.3f}, {far[2]:.3f}] | {n_answerable} |",
        f"| correct abstention (unanswerable) | {point.correct_abstention_rate:.3f} "
        f"| [{car[1]:.3f}, {car[2]:.3f}] | {n_unanswerable} |",
        "",
        "Per-type correct abstention: "
        + ", ".join(f"type {key}: {value:.2f}" for key, value in point.per_type.items()),
    ]
    (out_dir / "abstention_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_test(config: PipelineConfig, args: argparse.Namespace) -> int:
    """Report abstention on the test split with the config's fixed threshold."""
    if not args.i_know_this_is_final:
        print(
            "refusing to evaluate the test split without --i-know-this-is-final\n"
            "test is for final reporting only (docs/08-evaluation-protocol.md)",
            file=sys.stderr,
        )
        return 2

    threshold = args.threshold
    if threshold is None:
        threshold = config.abstain.params.get("threshold")
    if threshold is None:
        print(
            "--split test requires a threshold: set abstain.params.threshold in the "
            "config or pass --threshold",
            file=sys.stderr,
        )
        return 2
    threshold = float(threshold)

    queries, documents = load_benchmark_split(
        benchmark=config.data.benchmark,
        mini_limit=config.data.mini_limit,
        read_mode=config.data.read_mode,
        split="test",
    )
    if args.limit:
        queries = queries[: args.limit]
    negatives = [n for n in load_negatives(NEGATIVES_PATH) if n.split == "test"]
    print(f"answerable test queries: {len(queries)}; unanswerable test: {len(negatives)}")

    point, intervals = evaluate_fixed_threshold(
        config, queries, negatives, documents, threshold
    )
    out_dir = args.out_dir or ARTIFACTS_DIR / "abstention-test"
    write_test_artefacts(
        point,
        intervals,
        out_dir,
        split="test",
        n_answerable=len(queries),
        n_unanswerable=len(negatives),
    )

    run_id = (
        f"{config.name}-abstention-test-{config.hash()}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    )
    log_test_event(run_id=run_id, config=config, reason=args.reason)

    far = intervals["false_abstention_rate"]
    car = intervals["correct_abstention_rate"]
    print()
    print(f"threshold: {threshold:.4f} (fixed, not selected on test)")
    print(
        f"false abstention:   {point.false_abstention_rate:.3f} "
        f"[{far[1]:.3f}, {far[2]:.3f}] n={len(queries)}"
    )
    print(
        f"correct abstention: {point.correct_abstention_rate:.3f} "
        f"[{car[1]:.3f}, {car[2]:.3f}] n={len(negatives)}"
    )
    print(f"per type: {point.per_type}")
    print(f"\nwrote {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="fixed threshold for --split test; defaults to the config's abstain threshold",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--i-know-this-is-final",
        action="store_true",
        help="required for --split test; the run is recorded in logs/test_runs.log",
    )
    parser.add_argument("--reason", default="final abstention evaluation")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.split == "test":
        return _run_test(config, args)

    queries, _ = load_benchmark_split(
        benchmark=config.data.benchmark,
        mini_limit=config.data.mini_limit,
        read_mode=config.data.read_mode,
        split="train",
    )
    if args.limit:
        queries = queries[: args.limit]

    negatives = [n for n in load_negatives(NEGATIVES_PATH) if n.split == "train"]
    print(f"answerable train queries: {len(queries)}; unanswerable train: {len(negatives)}")

    samples = collect_samples(config, queries, negatives)
    fold_counts = np.unique([s.fold for s in samples], return_counts=True)
    print(f"collected {len(samples)} samples; folds {dict(zip(*fold_counts, strict=True))}")

    curves: dict[str, list[OperatingPoint]] = {}
    operating: dict[str, OperatingPoint] = {}
    nested: dict[str, OperatingPoint] = {}
    intervals: dict[str, dict[str, tuple[float, float, float]]] = {}

    for policy in POLICIES:
        confidences = confidences_for(policy, samples, args.folds, config.seed)
        points = sweep(policy, confidences, samples)
        curves[policy] = points
        operating[policy] = choose_operating_point(points, args.target)

        decisions, thresholds = nested_decisions(
            policy, confidences, samples, args.folds, args.target
        )
        nested[policy] = summarise(decisions, float(np.mean(thresholds)) if thresholds else 0.0)
        intervals[policy] = operating_point_ci(decisions, seed=config.seed)

    out_dir = args.out_dir or ARTIFACTS_DIR / "abstention"
    write_artefacts(curves, operating, out_dir, nested=nested, intervals=intervals)

    print()
    print(f"{'policy':12s} {'thresh':>8s} {'false':>7s} {'correct':>8s} {'prec':>7s} {'f1':>7s}")
    for policy, point in operating.items():
        print(
            f"{policy:12s} {point.threshold:8.4f} {point.false_abstention_rate:7.3f} "
            f"{point.correct_abstention_rate:8.3f} {point.abstention_precision:7.3f} "
            f"{point.abstention_f1:7.3f}"
        )
        print(f"             per type: {point.per_type}")

    print("\nnested (threshold chosen on other folds, applied to held-out fold):")
    print(
        f"{'policy':12s} {'false':>7s} {'95% CI':>16s} "
        f"{'correct':>8s} {'95% CI':>16s} {'f1':>7s}"
    )
    for policy, point in nested.items():
        far_ci = intervals[policy]["false_abstention_rate"]
        car_ci = intervals[policy]["correct_abstention_rate"]
        print(
            f"{policy:12s} {point.false_abstention_rate:7.3f} "
            f"[{far_ci[1]:.3f},{far_ci[2]:.3f}] "
            f"{point.correct_abstention_rate:8.3f} [{car_ci[1]:.3f},{car_ci[2]:.3f}] "
            f"{point.abstention_f1:7.3f}"
        )

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
