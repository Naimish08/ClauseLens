"""Metrics. This module is the single source of truth for every score.

The definitions follow LegalBench-RAG (Pipitone & Houir Alami, 2024) and its
reference implementation in ``legalbenchrag/run_benchmark.py``. See
``docs/03-metrics.md`` for the reasoning and the tests for hand-computed
fixtures.

Two conventions worth remembering:

* Everything is macro-averaged over queries. A long answer does not outvote a
  short one.
* ``mode="paper"`` sums per-retrieved-snippet lengths, exactly like the reference
  implementation. ``mode="union"`` merges overlapping spans first. They differ
  only when retrieval returns overlapping chunks, and the difference is
  deliberate (ADR-005): the paper's definition can exceed 1.0 for recall in that
  case, which is a property of the definition, not an error here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.data import Query
from src.types import RetrievedSpan, contains, sort_and_merge_spans, sort_retrieved

MetricMode = Literal["paper", "union"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _intersection(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def harmonic_mean(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _rate(flags: Iterable[bool]) -> float:
    items = list(flags)
    return sum(1 for f in items if f) / len(items) if items else 0.0


# --------------------------------------------------------------------------- #
# Per-query scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class QueryScore:
    """Everything measured for one query at one ``k``."""

    query: str
    category: str
    split: str
    answerable: bool
    abstained: bool
    k: int | None
    n_retrieved: int
    retrieved_len: int
    overlap_len: int
    gt_len: int
    recall: float
    precision: float
    f1: float
    doc_hit: bool
    span_hit: bool
    chunk_hit: bool
    mrr: float

    @property
    def failure(self) -> str | None:
        """Coarse diagnosis, used by the improvement loop.

        Ordered by specificity: a span that is partially covered is a chunking
        problem, not a ranking problem, even though both look like "no match".
        """
        if self.abstained:
            return "false_abstention" if self.answerable else None
        if not self.answerable:
            return "missed_abstention"
        if self.recall >= 1.0:
            return "extra_context" if self.precision < 1.0 else None
        if not self.span_hit:
            return "wrong_section" if self.doc_hit else "wrong_document"
        return "partial_span"

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "category": self.category,
            "split": self.split,
            "answerable": self.answerable,
            "abstained": self.abstained,
            "k": self.k,
            "n_retrieved": self.n_retrieved,
            "retrieved_len": self.retrieved_len,
            "overlap_len": self.overlap_len,
            "gt_len": self.gt_len,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "doc_hit": self.doc_hit,
            "span_hit": self.span_hit,
            "chunk_hit": self.chunk_hit,
            "mrr": self.mrr,
            "failure": self.failure,
        }


def score_query(
    query: Query,
    retrieved: Sequence[RetrievedSpan],
    *,
    k: int | None = None,
    mode: MetricMode = "paper",
) -> QueryScore:
    """Score one query against its ground truth.

    ``retrieved`` is re-ranked with :func:`src.types.sort_retrieved` before ``k``
    is applied, so a retriever cannot influence results through input order.
    """
    ranked = sort_retrieved(retrieved)
    if k is not None:
        ranked = ranked[:k]

    gt_by_file: dict[str, list[tuple[int, int]]] = {}
    for snippet in query.snippets:
        gt_by_file.setdefault(snippet.file_path, []).append(snippet.span)
    gt_len = sum(snippet.length for snippet in query.snippets)

    if mode == "paper":
        overlap_len = sum(
            _intersection(span.span, gt)
            for span in ranked
            for gt in gt_by_file.get(span.file_path, ())
        )
        retrieved_len = sum(span.length for span in ranked)
    elif mode == "union":
        retrieved_by_file: dict[str, list[tuple[int, int]]] = {}
        for span in ranked:
            retrieved_by_file.setdefault(span.file_path, []).append(span.span)
        overlap_len = 0
        retrieved_len = 0
        for file_path, spans in retrieved_by_file.items():
            merged_retrieved = sort_and_merge_spans(spans)
            merged_gt = sort_and_merge_spans(gt_by_file.get(file_path, ()))
            retrieved_len += sum(end - start for start, end in merged_retrieved)
            overlap_len += sum(
                _intersection(r, g) for r in merged_retrieved for g in merged_gt
            )
    else:
        raise ValueError(f"unknown metric mode: {mode!r}")

    span_hit = False
    chunk_hit = False
    mrr = 0.0
    for rank, span in enumerate(ranked, start=1):
        hits = [
            gt for gt in gt_by_file.get(span.file_path, ()) if _intersection(span.span, gt) > 0
        ]
        if hits:
            span_hit = True
            if mrr == 0.0:
                mrr = 1.0 / rank
            if any(contains(span.span, gt) for gt in hits):
                chunk_hit = True

    recall = overlap_len / gt_len if gt_len else 0.0
    precision = overlap_len / retrieved_len if retrieved_len else 0.0

    return QueryScore(
        query=query.text,
        category=query.category,
        split=query.split,
        answerable=gt_len > 0,
        abstained=not ranked,
        k=k,
        n_retrieved=len(ranked),
        retrieved_len=retrieved_len,
        overlap_len=overlap_len,
        gt_len=gt_len,
        recall=recall,
        precision=precision,
        f1=harmonic_mean(recall, precision),
        doc_hit=any(span.file_path in gt_by_file for span in ranked),
        span_hit=span_hit,
        chunk_hit=chunk_hit,
        mrr=mrr,
    )


def score_query_all_k(
    query: Query,
    retrieved: Sequence[RetrievedSpan],
    ks: Sequence[int],
    *,
    mode: MetricMode = "paper",
) -> list[QueryScore]:
    """Score the same ranking at several cut-offs (efficient: one sort)."""
    return [score_query(query, retrieved, k=k, mode=mode) for k in ks]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Aggregate:
    """Macro-averaged metrics over a set of queries."""

    n: int
    n_precision: int
    recall: float
    precision: float
    f1: float
    doc_hit: float
    span_hit: float
    chunk_hit: float
    mrr: float
    abstention_rate: float
    false_abstention_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "n_precision": self.n_precision,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "doc_hit": self.doc_hit,
            "span_hit": self.span_hit,
            "chunk_hit": self.chunk_hit,
            "mrr": self.mrr,
            "abstention_rate": self.abstention_rate,
            "false_abstention_rate": self.false_abstention_rate,
        }


EMPTY_AGGREGATE = Aggregate(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def aggregate(scores: Sequence[QueryScore]) -> Aggregate:
    """Macro-average a set of query scores.

    Retrieval metrics are computed over **answerable** queries only; abstention
    rates are computed over everything passed in. Precision's population excludes
    queries where the system abstained (ADR-006): abstaining is already penalised
    through ``recall`` and ``false_abstention_rate``.
    """
    if not scores:
        return EMPTY_AGGREGATE

    answerable = [s for s in scores if s.answerable]
    scored = [s for s in answerable if not s.abstained]

    recall = _mean([s.recall for s in answerable])
    precision = _mean([s.precision for s in scored])
    return Aggregate(
        n=len(scores),
        n_precision=len(scored),
        recall=recall,
        precision=precision,
        f1=harmonic_mean(recall, precision),
        doc_hit=_mean([float(s.doc_hit) for s in answerable]),
        span_hit=_mean([float(s.span_hit) for s in answerable]),
        chunk_hit=_mean([float(s.chunk_hit) for s in answerable]),
        mrr=_mean([s.mrr for s in answerable]),
        abstention_rate=_rate([s.abstained for s in scores]),
        false_abstention_rate=_rate([s.abstained for s in answerable]),
    )


def aggregate_by_category(scores: Sequence[QueryScore]) -> dict[str, Aggregate]:
    """Per-clause-type breakdown. Always report ``n`` alongside these."""
    grouped: dict[str, list[QueryScore]] = {}
    for score in scores:
        grouped.setdefault(score.category, []).append(score)
    return {category: aggregate(group) for category, group in sorted(grouped.items())}


def aggregate_by_split(scores: Sequence[QueryScore]) -> dict[str, Aggregate]:
    grouped: dict[str, list[QueryScore]] = {}
    for score in scores:
        grouped.setdefault(score.split, []).append(score)
    return {split: aggregate(group) for split, group in sorted(grouped.items())}


def metric_vector(scores: Sequence[QueryScore], metric: str) -> list[float]:
    """Per-query values for bootstrapping. ``metric`` names a field on QueryScore."""
    values = []
    for score in scores:
        value = getattr(score, metric, None)
        if value is None:
            raise AttributeError(f"no such metric: {metric!r}")
        values.append(float(value))
    return values


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AbstentionSummary:
    """Abstention quality in both directions."""

    n_answerable: int
    n_unanswerable: int
    false_abstention_rate: float
    correct_abstention_rate: float
    abstention_precision: float
    abstention_f1: float
    per_type: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "false_abstention_rate": self.false_abstention_rate,
            "correct_abstention_rate": self.correct_abstention_rate,
            "abstention_precision": self.abstention_precision,
            "abstention_f1": self.abstention_f1,
            "per_type": self.per_type,
        }


def abstention_summary(
    answerable_abstained: Sequence[bool],
    unanswerable_abstained: Sequence[bool],
    unanswerable_types: Sequence[str] | None = None,
) -> AbstentionSummary:
    """Summarise abstention decisions on answerable and unanswerable queries."""
    if unanswerable_types is not None and len(unanswerable_types) != len(unanswerable_abstained):
        raise ValueError("unanswerable_types must align with unanswerable_abstained")

    n_a = len(answerable_abstained)
    n_u = len(unanswerable_abstained)
    false_abstentions = sum(1 for flag in answerable_abstained if flag)
    correct_abstentions = sum(1 for flag in unanswerable_abstained if flag)

    far = false_abstentions / n_a if n_a else 0.0
    car = correct_abstentions / n_u if n_u else 0.0
    total_abstentions = false_abstentions + correct_abstentions
    precision = correct_abstentions / total_abstentions if total_abstentions else 0.0

    per_type: dict[str, float] = {}
    if unanswerable_types is not None:
        buckets: dict[str, list[bool]] = {}
        for label, flag in zip(unanswerable_types, unanswerable_abstained, strict=True):
            buckets.setdefault(label, []).append(flag)
        per_type = {label: _rate(flags) for label, flags in sorted(buckets.items())}

    return AbstentionSummary(
        n_answerable=n_a,
        n_unanswerable=n_u,
        false_abstention_rate=far,
        correct_abstention_rate=car,
        abstention_precision=precision,
        abstention_f1=harmonic_mean(1.0 - far, car),
        per_type=per_type,
    )


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Mean difference between two systems and its percentile interval."""

    delta: float
    lo: float
    hi: float
    n: int
    n_resamples: int
    significant: bool

    def __str__(self) -> str:
        return f"{self.delta:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}] n={self.n}"


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 13,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired bootstrap on the mean of ``a - b`` (same queries, two systems).

    ``significant`` is True when the interval excludes zero. It is a convenience
    flag for the loop's gate, not a substitute for reading the interval.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired series must match: {x.shape} vs {y.shape}")
    n = x.size
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0, 0, False)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    deltas = x[indices].mean(axis=1) - y[indices].mean(axis=1)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        delta=float(x.mean() - y.mean()),
        lo=float(lo),
        hi=float(hi),
        n=n,
        n_resamples=n_resamples,
        significant=bool(lo > 0 or hi < 0),
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 13,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean and percentile interval for one series (e.g. a per-category mean)."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(n_resamples, data.size))
    means = data[indices].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(data.mean()), float(lo), float(hi))
