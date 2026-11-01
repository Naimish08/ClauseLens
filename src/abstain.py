"""Abstention policies.

An abstainer decides whether to return nothing for a query. It sees only retrieval
output — never ground truth. The four policies in ``docs/06-abstention.md`` share
one tiny interface so the evaluation harness can sweep them without special cases.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.types import RetrievedSpan


@dataclass(frozen=True, slots=True)
class AbstentionDecision:
    abstain: bool
    reason: str
    score: float


class Abstainer(Protocol):
    name: str

    def decide(self, retrieved: Sequence[RetrievedSpan]) -> AbstentionDecision: ...


@dataclass(frozen=True, slots=True)
class NeverAbstain:
    """Baseline: return whatever retrieval produced, even nothing."""

    name: str = "none"

    def decide(self, retrieved: Sequence[RetrievedSpan]) -> AbstentionDecision:
        if not retrieved:
            return AbstentionDecision(True, "no results", 0.0)
        return AbstentionDecision(False, "returning results", retrieved[0].score)


@dataclass(frozen=True, slots=True)
class ThresholdAbstainer:
    """Abstain when the top score is below ``threshold``.

    The naive policy the project is arguing against: scores are not comparable
    across retrievers or query types, so one global number cannot be right.
    """

    threshold: float = 0.0
    name: str = "threshold"

    def decide(self, retrieved: Sequence[RetrievedSpan]) -> AbstentionDecision:
        if not retrieved:
            return AbstentionDecision(True, "no results", 0.0)
        top = retrieved[0].score
        if top < self.threshold:
            return AbstentionDecision(
                True, f"top score {top:.4f} < threshold {self.threshold:.4f}", top
            )
        return AbstentionDecision(False, "above threshold", top)


ABSTAINERS: dict[str, Callable[..., Abstainer]] = {
    "none": NeverAbstain,
    "threshold": ThresholdAbstainer,
}


def build_abstainer(name: str, params: dict[str, Any] | None = None) -> Abstainer:
    if name not in ABSTAINERS:
        raise ValueError(f"unknown abstainer {name!r}; available: {sorted(ABSTAINERS)}")
    return ABSTAINERS[name](**(params or {}))
