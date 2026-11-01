"""Failure diagnosis for the improvement loop.

Every failed query gets exactly one category, in a fixed precedence order, and is
weighted by how much recall it lost. Weighting matters: fixing ten near-complete
queries is worth less than fixing one total miss.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.metrics import QueryScore

CATEGORIES: tuple[str, ...] = (
    "wrong_document",
    "wrong_section",
    "partial_span",
    "extra_context",
    "false_abstention",
    "missed_abstention",
)


@dataclass
class Diagnosis:
    counts: dict[str, int] = field(default_factory=dict)
    loss: dict[str, float] = field(default_factory=dict)

    @property
    def total_loss(self) -> float:
        return sum(self.loss.values())

    @property
    def dominant(self) -> str | None:
        if not self.loss:
            return None
        return max(self.loss, key=lambda category: self.loss[category])

    def share(self, category: str) -> float:
        total = self.total_loss
        return self.loss.get(category, 0.0) / total if total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "dominant_failure": self.dominant,
            "counts": dict(self.counts),
            "loss": {key: round(value, 3) for key, value in self.loss.items()},
            "loss_share": {
                key: round(self.share(key), 4)
                for key in sorted(self.loss, key=self.loss.get, reverse=True)
            },
        }


def diagnose(scores: Sequence[QueryScore]) -> Diagnosis:
    """Count failures by category and attribute recall loss to each."""
    diagnosis = Diagnosis(counts={category: 0 for category in CATEGORIES})
    for score in scores:
        category = score.failure
        if category is None:
            continue
        diagnosis.counts[category] = diagnosis.counts.get(category, 0) + 1
        diagnosis.loss[category] = diagnosis.loss.get(category, 0.0) + (1.0 - score.recall)
    return diagnosis
