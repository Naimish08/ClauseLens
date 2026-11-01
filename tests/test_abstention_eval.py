"""Tests for the abstention evaluation protocol."""

from __future__ import annotations

import numpy as np

from eval.run_abstention import (
    POLICIES,
    Sample,
    choose_operating_point,
    evaluate_threshold,
    nested_decisions,
    summarise,
    sweep,
)
from src.abstain import QueryFeatures
from src.data import fold_of


def sample(
    top1: float,
    *,
    answerable: bool,
    type_id: int | None = None,
    fold: int | None = None,
) -> Sample:
    query = "q"
    features = QueryFeatures(
        query=query,
        category="c",
        top1=top1,
        top2=0.0,
        gap=top1,
        relative_gap=1.0,
        n_results=1,
        retrieved_len=10,
        query_len=len(query),
    )
    return Sample(
        features=features,
        answerable=answerable,
        negative_type=type_id,
        fold=fold_of(query) if fold is None else fold,
    )


def test_evaluate_threshold_counts_both_directions() -> None:
    samples = [
        sample(10.0, answerable=True),
        sample(2.0, answerable=True),
        sample(1.0, answerable=False, type_id=1),
        sample(5.0, answerable=False, type_id=3),
    ]
    confidences = [s.features.top1 for s in samples]
    point = evaluate_threshold(confidences, samples, threshold=3.0)
    assert point.false_abstention_rate == 0.5  # the 2.0 query
    assert point.correct_abstention_rate == 0.5  # the 1.0 negative
    assert point.abstention_precision == 0.5
    assert point.per_type == {"1": 1.0, "3": 0.0}


def test_sweep_is_monotone_in_abstention() -> None:
    samples = [
        sample(10.0, answerable=True),
        sample(1.0, answerable=True),
        sample(0.5, answerable=False, type_id=2),
        sample(5.0, answerable=False, type_id=2),
    ]
    points = sweep("threshold", [s.features.top1 for s in samples], samples, n_points=10)
    car = [p.correct_abstention_rate for p in points]
    far = [p.false_abstention_rate for p in points]
    assert car == sorted(car)
    assert far == sorted(far)


def test_choose_operating_point_respects_the_target() -> None:
    samples = [
        sample(10.0, answerable=True),
        sample(1.0, answerable=True),
        sample(0.5, answerable=False, type_id=1),
    ]
    points = sweep("threshold", [s.features.top1 for s in samples], samples)
    chosen = choose_operating_point(points, target=1.0)
    assert chosen.correct_abstention_rate >= 1.0


def test_choose_operating_point_falls_back_when_target_unreachable() -> None:
    samples = [sample(10.0, answerable=True), sample(9.0, answerable=False, type_id=1)]
    points = sweep("threshold", [s.features.top1 for s in samples], samples)
    chosen = choose_operating_point(points, target=1.1)
    assert chosen.abstention_f1 == max(p.abstention_f1 for p in points)


def test_nested_decisions_covers_every_sample() -> None:
    samples = [
        sample(10.0, answerable=True, fold=0),
        sample(8.0, answerable=True, fold=1),
        sample(1.0, answerable=False, type_id=1, fold=2),
    ]
    decisions, thresholds = nested_decisions(
        "threshold", [s.features.top1 for s in samples], samples, n_folds=5, target=0.5
    )
    assert len(decisions) == len(samples)
    assert thresholds


def test_summarise_matches_evaluate_threshold() -> None:
    samples = [
        sample(10.0, answerable=True),
        sample(2.0, answerable=True),
        sample(1.0, answerable=False, type_id=4),
    ]
    confidences = [s.features.top1 for s in samples]
    direct = evaluate_threshold(confidences, samples, 3.0)
    from eval.run_abstention import Decision

    decisions = [
        Decision(abstain=c < 3.0, answerable=s.answerable, negative_type=s.negative_type)
        for c, s in zip(confidences, samples, strict=True)
    ]
    summarised = summarise(decisions, 3.0)
    assert summarised.false_abstention_rate == direct.false_abstention_rate
    assert summarised.correct_abstention_rate == direct.correct_abstention_rate


def test_policy_list_is_complete() -> None:
    assert set(POLICIES) == {"threshold", "margin", "category", "classifier"}


def test_thresholds_from_sweep_are_finite() -> None:
    samples = [sample(v, answerable=(i % 2 == 0)) for i, v in enumerate(np.linspace(0, 5, 6))]
    points = sweep("threshold", [s.features.top1 for s in samples], samples)
    assert all(np.isfinite(p.threshold) for p in points)
