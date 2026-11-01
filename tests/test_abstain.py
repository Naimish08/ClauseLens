"""Tests for abstention policies and the feature contract."""

from __future__ import annotations

import numpy as np
import pytest

from src.abstain import (
    FEATURE_NAMES,
    CategoryThresholdAbstainer,
    LogisticAbstainer,
    MarginAbstainer,
    NeverAbstain,
    QueryFeatures,
    ThresholdAbstainer,
    build_abstainer,
)
from src.types import RetrievedSpan


def spans(*scores: float) -> list[RetrievedSpan]:
    return [
        RetrievedSpan("cuad/a.txt", i * 10, i * 10 + 10, score) for i, score in enumerate(scores)
    ]


def features(
    query: str = "Consider X; What is the governing law?", *scores: float
) -> QueryFeatures:
    return QueryFeatures.from_retrieval(query, spans(*scores))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_features_from_retrieval() -> None:
    f = features("Consider X; What is the governing law?", 9.0, 4.0, 1.0)
    assert f.top1 == 9.0 and f.top2 == 4.0
    assert f.gap == 5.0
    assert f.relative_gap == pytest.approx(5.0 / 9.0)
    assert f.n_results == 3
    assert f.retrieved_len == 30
    assert f.category == "What is the governing law?"


def test_features_with_no_results() -> None:
    f = QueryFeatures.from_retrieval("q", [])
    assert f.top1 == 0.0 and f.gap == 0.0 and f.n_results == 0


def test_features_are_ranked_independently_of_input_order() -> None:
    a = QueryFeatures.from_retrieval("q", spans(1.0, 9.0))
    b = QueryFeatures.from_retrieval("q", list(reversed(spans(1.0, 9.0))))
    assert a.top1 == b.top1 == 9.0


def test_query_length_is_not_a_model_feature() -> None:
    """Regression guard: query length separated the classes by construction.

    The authored negatives are shorter than benchmark queries, so including length
    let a classifier look excellent for the wrong reason (docs/06-abstention.md).
    """
    assert "query_len" not in FEATURE_NAMES
    assert features("a" * 50, 1.0, 0.5).vector().shape == (len(FEATURE_NAMES),)


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def test_never_abstains_when_results_exist() -> None:
    policy = NeverAbstain()
    assert policy.decide(features("q", 1.0)).abstain is False
    assert policy.decide(features("q")).abstain is True


def test_threshold_policy() -> None:
    policy = ThresholdAbstainer(threshold=5.0)
    assert policy.decide(features("q", 4.9)).abstain is True
    assert policy.decide(features("q", 5.0)).abstain is False
    assert policy.decide(features("q", 9.0)).abstain is False


def test_threshold_policy_handles_no_results() -> None:
    decision = ThresholdAbstainer(threshold=0.0).decide(features("q"))
    assert decision.abstain and decision.reason == "no results"


def test_margin_policy_uses_the_gap() -> None:
    policy = MarginAbstainer(margin=2.0)
    assert policy.confidence(features("q", 4.0, 3.0)) == pytest.approx(1.0)
    assert policy.decide(features("q", 4.0, 3.0)).abstain is True
    assert policy.decide(features("q", 9.0, 1.0)).abstain is False


def test_margin_policy_with_a_single_result_uses_zero_gap() -> None:
    assert MarginAbstainer(margin=0.1).decide(features("q", 9.0)).abstain is True


def test_category_policy_normalises_by_scale() -> None:
    policy = CategoryThresholdAbstainer(
        threshold=0.5,
        scales={"What is the governing law?": 10.0},
        default_scale=2.0,
    )
    # 6 / 10 = 0.6 -> accepted; 6 / 2 = 3.0 -> accepted for an unknown category.
    confident = features("Consider X; What is the governing law?", 6.0)
    assert policy.confidence(confident) == pytest.approx(0.6)
    assert policy.decide(features("Consider X; What is the governing law?", 4.0)).abstain is True
    assert policy.confidence(features("Consider X; Other clause?", 4.0)) == pytest.approx(2.0)


def test_category_policy_falls_back_to_default_scale() -> None:
    policy = CategoryThresholdAbstainer(threshold=0.5, scales={}, default_scale=4.0)
    assert policy.confidence(features("X; unknown type", 2.0)) == pytest.approx(0.5)


class _StubModel:
    def __init__(self, probability_unanswerable: float) -> None:
        self.probability = probability_unanswerable
        self.seen: list[np.ndarray] = []

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.seen.append(x)
        return np.array([[1 - self.probability, self.probability]])


def test_logistic_policy_maps_probability_to_confidence() -> None:
    model = _StubModel(probability_unanswerable=0.9)
    policy = LogisticAbstainer(model=model, threshold=0.5)
    assert policy.confidence(features("q", 1.0)) == pytest.approx(0.1)
    assert policy.decide(features("q", 1.0)).abstain is True
    assert model.seen[0].shape == (1, len(FEATURE_NAMES))


def test_logistic_policy_accepts_when_answerable_probability_is_high() -> None:
    policy = LogisticAbstainer(model=_StubModel(0.1), threshold=0.5)
    assert policy.decide(features("q", 1.0)).abstain is False


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_builds_known_policies() -> None:
    assert build_abstainer("none").name == "none"
    assert build_abstainer("threshold", {"threshold": 1.0}).name == "threshold"
    assert build_abstainer("margin", {"margin": 1.0}).name == "margin"
    assert build_abstainer("category", {"threshold": 1.0}).name == "category"


def test_registry_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unknown abstainer"):
        build_abstainer("oracle")
