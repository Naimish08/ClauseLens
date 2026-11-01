"""Tests for the metric implementation.

Every headline number in the report goes through this module, so the fixtures
here are hand-computed rather than property-based. If a definition changes, one of
these tests must change with it — deliberately.
"""

from __future__ import annotations

import pytest

from src.data import Query, Snippet
from src.metrics import (
    abstention_summary,
    aggregate,
    aggregate_by_category,
    aggregate_by_split,
    bootstrap_ci,
    harmonic_mean,
    metric_vector,
    paired_bootstrap,
    score_query,
)
from src.types import RetrievedSpan, contains, make_chunk, sort_and_merge_spans, sort_retrieved


def q(*spans: tuple[int, int], text: str = "q", file_path: str = "doc.txt") -> Query:
    return Query(
        text=text,
        snippets=tuple(
            Snippet(file_path=file_path, start=start, end=end, answer="") for start, end in spans
        ),
    )


def r(start: int, end: int, *, score: float = 1.0, file_path: str = "doc.txt") -> RetrievedSpan:
    return RetrievedSpan(file_path=file_path, start=start, end=end, score=score)


# --------------------------------------------------------------------------- #
# Core precision / recall
# --------------------------------------------------------------------------- #
def test_perfect_retrieval_scores_one() -> None:
    score = score_query(q((2, 5)), [r(2, 5)])
    assert (score.recall, score.precision, score.f1) == (1.0, 1.0, 1.0)
    assert score.doc_hit and score.span_hit and score.chunk_hit
    assert score.mrr == 1.0
    assert score.failure is None


def test_half_of_the_ground_truth() -> None:
    score = score_query(q((0, 10)), [r(0, 5)])
    assert score.recall == 0.5
    assert score.precision == 1.0
    assert score.span_hit
    assert not score.chunk_hit
    assert score.failure == "partial_span"


def test_ground_truth_plus_equal_extra_text() -> None:
    score = score_query(q((0, 10)), [r(0, 20)])
    assert score.recall == 1.0
    assert score.precision == 0.5
    assert score.chunk_hit
    assert score.failure == "extra_context"


def test_two_ground_truth_spans_retrieve_one() -> None:
    score = score_query(q((0, 5), (10, 15)), [r(0, 5)])
    assert score.recall == 0.5
    assert score.precision == 1.0
    assert score.chunk_hit


def test_multi_document_ground_truth() -> None:
    query = Query(
        text="q",
        snippets=(
            Snippet("a.txt", 0, 10, ""),
            Snippet("b.txt", 0, 10, ""),
        ),
    )
    score = score_query(query, [RetrievedSpan("a.txt", 0, 10, 1.0)])
    assert score.recall == 0.5
    assert score.precision == 1.0
    assert score.doc_hit


def test_wrong_document_is_diagnosed() -> None:
    score = score_query(q((0, 10)), [r(50, 60, file_path="other.txt")])
    assert score.recall == 0.0 and score.precision == 0.0
    assert not score.doc_hit
    assert score.failure == "wrong_document"


def test_right_document_wrong_section_is_diagnosed() -> None:
    score = score_query(q((0, 10)), [r(50, 60)])
    assert score.doc_hit
    assert not score.span_hit
    assert score.failure == "wrong_section"


def test_empty_retrieval_abstains_without_error() -> None:
    score = score_query(q((0, 10)), [])
    assert score.abstained
    assert score.recall == 0.0 and score.precision == 0.0
    assert score.failure == "false_abstention"


def test_unanswerable_query_with_results_is_missed_abstention() -> None:
    score = score_query(q(), [r(0, 10)])
    assert not score.answerable
    assert score.failure == "missed_abstention"


def test_unanswerable_query_with_no_results_is_not_a_failure() -> None:
    score = score_query(q(), [])
    assert score.failure is None


# --------------------------------------------------------------------------- #
# The paper/union divergence (ADR-005)
# --------------------------------------------------------------------------- #
def test_overlapping_retrieved_spans_change_recall_between_modes() -> None:
    """With overlapping chunks the paper definition double-counts characters.

    GT is 100 chars. Two retrieved chunks of 60 chars overlap by 20, so together
    they cover the GT exactly once. The paper definition sums per-snippet overlap
    (60 + 60 = 120) against a 100-char GT, giving recall 1.2 — above one. The
    union definition gives 1.0. This test exists to document that, not to bless it.
    """
    query = q((0, 100))
    retrieved = [r(0, 60), r(40, 100)]

    paper = score_query(query, retrieved, mode="paper")
    union = score_query(query, retrieved, mode="union")

    assert paper.recall == pytest.approx(1.2)
    assert paper.precision == pytest.approx(1.0)
    assert paper.retrieved_len == 120

    assert union.recall == pytest.approx(1.0)
    assert union.precision == pytest.approx(1.0)
    assert union.retrieved_len == 100


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown metric mode"):
        score_query(q((0, 1)), [r(0, 1)], mode="cosine")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Ranking, k and determinism
# --------------------------------------------------------------------------- #
def test_k_cuts_after_ranking_by_score() -> None:
    retrieved = [r(0, 10, score=0.1), r(50, 60, score=0.9), r(70, 80, score=0.5)]
    at_one = score_query(q((50, 60)), retrieved, k=1)
    at_two = score_query(q((50, 60)), retrieved, k=2)

    assert at_one.abstained is False
    assert at_one.span_hit and at_one.mrr == 1.0
    assert at_two.mrr == 1.0
    assert at_two.n_retrieved == 2


def test_mrr_uses_the_first_hit_rank() -> None:
    retrieved = [r(100, 110, score=3.0), r(0, 10, score=2.0), r(20, 30, score=1.0)]
    score = score_query(q((0, 10)), retrieved)
    assert score.mrr == 0.5


def test_input_order_does_not_affect_results() -> None:
    retrieved = [r(0, 10, score=0.2), r(20, 30, score=0.8)]
    a = score_query(q((0, 10)), retrieved)
    b = score_query(q((0, 10)), reversed(retrieved))
    assert a.as_dict() == b.as_dict()


def test_score_query_all_k_shares_the_ranking() -> None:
    retrieved = [r(20, 30, score=9.0), r(0, 10, score=8.0)]
    scores = [
        score_query(q((0, 10)), retrieved, k=k) for k in (1, 2)
    ]
    assert scores[0].span_hit is False
    assert scores[1].span_hit is True


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _perfect_and_abstained() -> list:
    return [
        score_query(q((0, 10)), [r(0, 10)]),
        score_query(q((0, 10)), []),
    ]


def test_aggregate_excludes_abstentions_from_precision() -> None:
    agg = aggregate(_perfect_and_abstained())
    assert agg.n == 2
    assert agg.recall == 0.5  # abstention contributes zero recall
    assert agg.precision == 1.0  # only the non-abstained query counts
    assert agg.n_precision == 1
    assert agg.false_abstention_rate == 0.5


def test_aggregate_is_macro_average() -> None:
    long_answer = score_query(q((0, 1000)), [r(0, 1000)])
    short_answer = score_query(q((0, 10)), [r(0, 10)])
    agg = aggregate([long_answer, short_answer])
    assert agg.recall == 1.0
    assert agg.n == 2


def test_aggregate_of_nothing_is_empty_not_an_error() -> None:
    agg = aggregate([])
    assert agg.n == 0 and agg.recall == 0.0


def test_aggregate_by_category_and_split() -> None:
    scores = [
        score_query(q((0, 10), text="Consider X; What is the governing law?"), [r(0, 10)]),
        score_query(q((0, 10), text="Consider X; What is the governing law?"), []),
        score_query(q((0, 10), text="Consider X; Is there an audit right?"), [r(0, 10)]),
    ]
    by_cat = aggregate_by_category(scores)
    assert set(by_cat) == {"Is there an audit right?", "What is the governing law?"}
    assert by_cat["Is there an audit right?"].n == 1
    assert by_cat["What is the governing law?"].n == 2
    assert by_cat["What is the governing law?"].recall == 0.5

    by_split = aggregate_by_split(scores)
    assert sum(a.n for a in by_split.values()) == 3


def test_metric_vector() -> None:
    scores = _perfect_and_abstained()
    assert metric_vector(scores, "recall") == [1.0, 0.0]
    with pytest.raises(AttributeError):
        metric_vector(scores, "nonsense")


def test_harmonic_mean_edge_case() -> None:
    assert harmonic_mean(0.0, 0.0) == 0.0
    assert harmonic_mean(1.0, 0.5) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #
def test_abstention_summary_both_directions() -> None:
    summary = abstention_summary(
        answerable_abstained=[False, False, False, True],
        unanswerable_abstained=[True, True, True, False, True],
        unanswerable_types=["3", "3", "4", "4", "1"],
    )
    assert summary.n_answerable == 4
    assert summary.n_unanswerable == 5
    assert summary.false_abstention_rate == pytest.approx(0.25)
    assert summary.correct_abstention_rate == pytest.approx(0.8)
    # 4 correct out of 5 total abstentions
    assert summary.abstention_precision == pytest.approx(0.8)
    assert summary.abstention_f1 == pytest.approx(2 * 0.75 * 0.8 / 1.55)
    assert summary.per_type == {"1": 1.0, "3": 1.0, "4": 0.5}


def test_abstention_summary_without_types() -> None:
    summary = abstention_summary([], [True, False])
    assert summary.per_type == {}
    assert summary.false_abstention_rate == 0.0
    assert summary.correct_abstention_rate == 0.5


def test_abstention_summary_rejects_misaligned_types() -> None:
    with pytest.raises(ValueError, match="align"):
        abstention_summary([], [True, False], ["3"])


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_identical_series_is_zero() -> None:
    values = [0.0, 0.5, 1.0, 0.25, 0.75]
    result = paired_bootstrap(values, values)
    assert result.delta == 0.0
    assert result.lo == 0.0 and result.hi == 0.0
    assert result.significant is False


def test_paired_bootstrap_detects_constant_offset() -> None:
    a = [x + 1.0 for x in (0.0, 0.5, 1.0, 0.25, 0.75)]
    b = [0.0, 0.5, 1.0, 0.25, 0.75]
    result = paired_bootstrap(a, b)
    assert result.delta == pytest.approx(1.0)
    assert result.lo == pytest.approx(1.0)
    assert result.hi == pytest.approx(1.0)
    assert result.significant


def test_paired_bootstrap_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_paired_bootstrap_is_seeded() -> None:
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    b = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_bootstrap_ci_of_empty_series_is_nan() -> None:
    mean, lo, hi = bootstrap_ci([])
    assert mean != mean and lo != lo and hi != hi  # NaN


# --------------------------------------------------------------------------- #
# Span utilities
# --------------------------------------------------------------------------- #
def test_sort_and_merge_spans() -> None:
    assert sort_and_merge_spans([(10, 20), (0, 5)]) == [(0, 5), (10, 20)]
    assert sort_and_merge_spans([(0, 10), (5, 15)]) == [(0, 15)]
    assert sort_and_merge_spans([(0, 10), (11, 15)], max_bridge_gap_len=1) == [(0, 15)]
    assert sort_and_merge_spans([]) == []


def test_contains() -> None:
    assert contains((0, 10), (2, 8))
    assert contains((0, 10), (0, 10))
    assert not contains((0, 10), (-1, 5))
    assert not contains((0, 10), (5, 11))


def test_sort_retrieved_breaks_ties_deterministically() -> None:
    spans = [
        RetrievedSpan("b.txt", 0, 5, 1.0),
        RetrievedSpan("a.txt", 10, 15, 1.0),
        RetrievedSpan("a.txt", 0, 5, 1.0),
    ]
    assert [s.chunk_id for s in sort_retrieved(spans)] == [""] * 3
    assert [(s.file_path, s.start) for s in sort_retrieved(spans)] == [
        ("a.txt", 0),
        ("a.txt", 10),
        ("b.txt", 0),
    ]


def test_make_chunk_id_is_canonical() -> None:
    chunk = make_chunk("cuad/a.txt", 5, 10, "hello")
    assert chunk.chunk_id == "cuad/a.txt:5:10"
    assert chunk.length == 5
