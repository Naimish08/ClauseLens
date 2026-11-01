"""Statistical acceptance for the improvement loop.

Three gates, all required (docs/07-improvement-loop.md):

1. effect size — mean recall@k improves by at least ``min_delta``
2. significance — paired bootstrap 95% interval on the delta excludes zero
3. consistency — the candidate wins on at least 4 of 5 folds
4. guardrails — precision, false abstention and latency do not degrade

"Consistency" is not a third significance test; it catches a gain concentrated in
one fold, which a pooled bootstrap can hide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.metrics import QueryScore, aggregate, paired_bootstrap

FOLD_WINS_REQUIRED = 4


@dataclass(frozen=True, slots=True)
class GateResult:
    accepted: bool
    effect_size: bool
    significance: bool
    consistency: bool
    guardrails: bool
    delta: float
    ci: tuple[float, float]
    fold_deltas: tuple[float, ...]
    notes: str

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "gate_effect_size": self.effect_size,
            "gate_significance": self.significance,
            "gate_consistency": self.consistency,
            "gate_guardrails": self.guardrails,
            "delta": round(self.delta, 4),
            "ci95": [round(self.ci[0], 4), round(self.ci[1], 4)],
            "fold_deltas": [round(value, 4) for value in self.fold_deltas],
            "notes": self.notes,
        }


def _recall_vector(scores: Sequence[QueryScore]) -> list[float]:
    return [score.recall for score in scores if score.answerable]


def _fold_means(scores: Sequence[QueryScore], folds: Sequence[int], n_folds: int) -> list[float]:
    buckets: list[list[float]] = [[] for _ in range(n_folds)]
    for score, fold in zip(scores, folds, strict=True):
        if score.answerable:
            buckets[fold].append(score.recall)
    return [sum(values) / len(values) if values else 0.0 for values in buckets]


def evaluate_gates(
    current: Sequence[QueryScore],
    candidate: Sequence[QueryScore],
    folds: Sequence[int],
    *,
    n_folds: int = 5,
    seed: int = 13,
    min_delta: float = 0.01,
    precision_relative_tolerance: float = 0.30,
    abstention_tolerance: float = 0.02,
    latency_current_p95: float = 0.0,
    latency_candidate_p95: float = 0.0,
    latency_factor: float = 1.5,
    latency_budget_ms: float = 250.0,
    baseline_mean_retrieved_len: float = 0.0,
    retrieved_len_factor: float = 2.5,
) -> GateResult:
    if len(current) != len(candidate):
        raise ValueError("current and candidate score lists must be aligned")

    current_recall = _recall_vector(current)
    candidate_recall = _recall_vector(candidate)
    bootstrap = paired_bootstrap(candidate_recall, current_recall, seed=seed)

    current_agg = aggregate(current)
    candidate_agg = aggregate(candidate)

    fold_current = _fold_means(current, folds, n_folds)
    fold_candidate = _fold_means(candidate, folds, n_folds)
    fold_deltas = tuple(c - b for c, b in zip(fold_candidate, fold_current, strict=True))
    wins = sum(1 for delta in fold_deltas if delta > 0)

    effect_size = bootstrap.delta >= min_delta
    significance = bootstrap.significant
    consistency = wins >= FOLD_WINS_REQUIRED

    # Relative, not absolute: precision near 0.01 makes an absolute tolerance of
    # 0.02 meaningless (it would permit a total collapse).
    precision_ok = candidate_agg.precision >= current_agg.precision * (
        1.0 - precision_relative_tolerance
    )

    # Guard against winning recall by returning ever more text. Measured against
    # the *baseline*, so doubling is not allowed to compound across iterations.
    mean_len = lambda scores: (  # noqa: E731
        sum(score.retrieved_len for score in scores if score.answerable)
        / max(1, sum(1 for score in scores if score.answerable))
    )
    current_mean_len = mean_len(current)
    candidate_mean_len = mean_len(candidate)
    reference_len = baseline_mean_retrieved_len or current_mean_len
    length_ok = candidate_mean_len <= reference_len * retrieved_len_factor
    abstention_ok = (
        candidate_agg.false_abstention_rate
        <= current_agg.false_abstention_rate + abstention_tolerance
    )
    # Absolute budget OR relative slack, whichever is larger. A purely relative
    # rule applied to a 3 ms baseline rejects any GPU component on principle
    # (ADR-018); a budget reflects what a user actually perceives.
    latency_ok = (
        latency_current_p95 <= 0.0
        or latency_candidate_p95
        <= max(latency_budget_ms, latency_current_p95 * latency_factor)
    )
    guardrails = precision_ok and abstention_ok and latency_ok and length_ok

    accepted = effect_size and significance and consistency and guardrails

    failing = []
    if not effect_size:
        failing.append(f"delta {bootstrap.delta:+.3f} < {min_delta}")
    if not significance:
        failing.append(f"CI includes 0 ({bootstrap.lo:+.3f}, {bootstrap.hi:+.3f})")
    if not consistency:
        failing.append(f"won {wins}/{n_folds} folds")
    if not guardrails:
        reasons = []
        if not precision_ok:
            reasons.append(
                f"precision {current_agg.precision:.3f}->{candidate_agg.precision:.3f}"
            )
        if not abstention_ok:
            reasons.append("false abstention rose")
        if not latency_ok:
            reasons.append(
                f"latency p95 {latency_current_p95:.1f}->{latency_candidate_p95:.1f} ms"
            )
        if not length_ok:
            reasons.append(
                f"mean retrieved {current_mean_len:.0f}->{candidate_mean_len:.0f} chars "
                f"(cap {reference_len * retrieved_len_factor:.0f})"
            )
        failing.append("guardrail: " + ", ".join(reasons))

    notes = "accepted" if accepted else "rejected: " + "; ".join(failing)

    return GateResult(
        accepted=accepted,
        effect_size=effect_size,
        significance=significance,
        consistency=consistency,
        guardrails=guardrails,
        delta=bootstrap.delta,
        ci=(bootstrap.lo, bootstrap.hi),
        fold_deltas=fold_deltas,
        notes=notes,
    )
