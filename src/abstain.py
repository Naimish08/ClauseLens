"""Abstention policies.

An abstainer sees only retrieval output — never ground truth — plus the query text.
Every policy exposes a single signed quantity, ``confidence``: the system abstains
when confidence is below a threshold. That common shape is what makes policies
comparable on the same trade-off curve.

Policies intentionally differ in what "confidence" means:

* ``threshold`` — the raw top score. Not comparable across retrievers.
* ``margin``    — the gap between rank 1 and rank 2.
* ``category``  — top score divided by a per-clause-type scale, so clause questions
  with systematically weaker matches are not punished.
* ``classifier`` — one minus a logistic model's probability that the query is
  unanswerable. Fitted offline (see ``eval/run_abstention.py``).

See ``docs/06-abstention.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from src.data import derive_category
from src.types import RetrievedSpan, sort_retrieved

#: Fixed column order for the classifier. Changing this invalidates fitted models.
#:
#: ``query_len`` is deliberately **excluded**: the authored negatives are shorter
#: than benchmark queries, so length separates the classes by construction and a
#: model would learn an authorship artefact rather than answerability
#: (docs/06-abstention.md).
FEATURE_NAMES: tuple[str, ...] = (
    "top1",
    "top2",
    "gap",
    "relative_gap",
    "log_top1",
    "n_results",
    "retrieved_len",
)


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    """Everything an abstainer may know about a query, computed from retrieval only."""

    query: str
    category: str
    top1: float
    top2: float
    gap: float
    relative_gap: float
    n_results: int
    retrieved_len: int
    query_len: int

    @classmethod
    def from_retrieval(cls, query: str, retrieved: Sequence[RetrievedSpan]) -> QueryFeatures:
        ranked = sort_retrieved(retrieved)
        scores = [span.score for span in ranked]
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0
        # A single result has no runner-up, so there is no evidence of decisiveness.
        gap = (top1 - top2) if len(scores) > 1 else 0.0
        return cls(
            query=query,
            category=derive_category(query),
            top1=top1,
            top2=top2,
            gap=gap,
            relative_gap=(gap / top1) if top1 > 0 else 0.0,
            n_results=len(ranked),
            retrieved_len=sum(span.length for span in ranked),
            query_len=len(query),
        )

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.top1,
                self.top2,
                self.gap,
                self.relative_gap,
                float(np.log1p(max(self.top1, 0.0))),
                self.n_results,
                self.retrieved_len,
            ],
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class AbstentionDecision:
    abstain: bool
    reason: str
    confidence: float


class Abstainer(Protocol):
    name: str

    def confidence(self, features: QueryFeatures) -> float: ...

    def decide(self, features: QueryFeatures) -> AbstentionDecision: ...


def _no_results(features: QueryFeatures) -> AbstentionDecision | None:
    if features.n_results == 0:
        return AbstentionDecision(True, "no results", 0.0)
    return None


@dataclass(frozen=True, slots=True)
class NeverAbstain:
    """Baseline: return whatever retrieval produced, even nothing."""

    name: str = "none"

    def confidence(self, features: QueryFeatures) -> float:
        return float("inf") if features.n_results else 0.0

    def decide(self, features: QueryFeatures) -> AbstentionDecision:
        empty = _no_results(features)
        if empty is not None:
            return empty
        return AbstentionDecision(False, "returning results", float("inf"))


@dataclass(frozen=True, slots=True)
class ThresholdAbstainer:
    """Abstain when the top score is below ``threshold`` — the strawman."""

    threshold: float = 0.0
    name: str = "threshold"

    def confidence(self, features: QueryFeatures) -> float:
        return features.top1

    def decide(self, features: QueryFeatures) -> AbstentionDecision:
        empty = _no_results(features)
        if empty is not None:
            return empty
        value = self.confidence(features)
        if value < self.threshold:
            return AbstentionDecision(
                True, f"top score {value:.4f} < {self.threshold:.4f}", value
            )
        return AbstentionDecision(False, "above threshold", value)


@dataclass(frozen=True, slots=True)
class MarginAbstainer:
    """Abstain when rank 1 and rank 2 are too close to distinguish."""

    margin: float = 0.0
    name: str = "margin"

    def confidence(self, features: QueryFeatures) -> float:
        return features.gap

    def decide(self, features: QueryFeatures) -> AbstentionDecision:
        empty = _no_results(features)
        if empty is not None:
            return empty
        value = self.confidence(features)
        if value < self.margin:
            return AbstentionDecision(True, f"score gap {value:.4f} < {self.margin:.4f}", value)
        return AbstentionDecision(False, "score gap is decisive", value)


@dataclass(frozen=True, slots=True)
class CategoryThresholdAbstainer:
    """Threshold on the top score, normalised by clause-type difficulty.

    ``scales`` maps a clause question type to the score scale that type tends to
    produce, so a single threshold works across categories with very different
    match strengths. Categories not in ``scales`` fall back to ``default_scale``.
    """

    threshold: float = 0.5
    scales: dict[str, float] = field(default_factory=dict)
    default_scale: float = 1.0
    name: str = "category"

    def confidence(self, features: QueryFeatures) -> float:
        scale = self.scales.get(features.category, self.default_scale)
        return features.top1 / scale if scale > 0 else features.top1

    def decide(self, features: QueryFeatures) -> AbstentionDecision:
        empty = _no_results(features)
        if empty is not None:
            return empty
        value = self.confidence(features)
        if value < self.threshold:
            return AbstentionDecision(
                True, f"normalised score {value:.4f} < {self.threshold:.4f}", value
            )
        return AbstentionDecision(False, "above normalised threshold", value)


@dataclass(frozen=True, slots=True)
class LogisticAbstainer:
    """Logistic regression on retrieval features; fitted outside the pipeline."""

    model: Any
    threshold: float = 0.5
    name: str = "classifier"

    def confidence(self, features: QueryFeatures) -> float:
        """Probability that the query is answerable."""
        probabilities = self.model.predict_proba(features.vector().reshape(1, -1))
        probability_unanswerable = float(probabilities[0, 1])
        return 1.0 - probability_unanswerable

    def decide(self, features: QueryFeatures) -> AbstentionDecision:
        empty = _no_results(features)
        if empty is not None:
            return empty
        value = self.confidence(features)
        if value < self.threshold:
            return AbstentionDecision(
                True, f"answerable probability {value:.4f} < {self.threshold:.4f}", value
            )
        return AbstentionDecision(False, "model accepts the query", value)


ABSTAINERS: dict[str, Callable[..., Abstainer]] = {
    "none": NeverAbstain,
    "threshold": ThresholdAbstainer,
    "margin": MarginAbstainer,
    "category": CategoryThresholdAbstainer,
}


def build_abstainer(name: str, params: dict[str, Any] | None = None) -> Abstainer:
    if name not in ABSTAINERS:
        raise ValueError(f"unknown abstainer {name!r}; available: {sorted(ABSTAINERS)}")
    return ABSTAINERS[name](**(params or {}))
