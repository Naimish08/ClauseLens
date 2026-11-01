"""Tests for the unanswerable set: schema, verification and the frozen file."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import (
    BENCHMARKS_DIR,
    Document,
    assign_split,
    load_benchmark,
    load_mini_cuad,
)
from src.negatives import (
    NegativeQuery,
    load_negatives,
    save_negatives,
    validate_negatives,
)

NEGATIVES_PATH = Path(__file__).parents[1] / "data" / "unanswerable.jsonl"
DATA_AVAILABLE = (BENCHMARKS_DIR / "cuad.json").exists()
SET_AVAILABLE = NEGATIVES_PATH.exists()


def _docs() -> dict[str, Document]:
    return {
        "cuad/a.txt": Document(
            "cuad/a.txt",
            "The Agreement contains an escrow of wind-down funds and a royalty of 2%.",
        ),
        "cuad/b.txt": Document("cuad/b.txt", "Either party may terminate for cause on breach."),
    }


def _entry(**overrides) -> NegativeQuery:
    defaults = dict(
        id="X1",
        query="Does the agreement require source code escrow?",
        type=3,
        rationale="test",
        scope="cuad/a.txt",
        absent_patterns=(r"source code escrow",),
        present_patterns=(),
        related_clause="",
        verified_by="test",
    )
    defaults.update(overrides)
    return NegativeQuery(**defaults)


# --------------------------------------------------------------------------- #
# Schema and verification logic
# --------------------------------------------------------------------------- #
def test_valid_entry_passes() -> None:
    report = validate_negatives([_entry()], _docs(), min_entries=1)
    assert report.ok
    assert report.per_type == {3: 1}


def test_absent_pattern_that_matches_is_a_violation() -> None:
    entry = _entry(absent_patterns=(r"escrow",))
    report = validate_negatives([entry], _docs(), min_entries=1)
    assert not report.ok
    assert "absent pattern present" in report.violations[0].problem


def test_corpus_scope_is_checked_across_all_documents() -> None:
    entry = _entry(scope="corpus", absent_patterns=(r"royalty",))
    report = validate_negatives([entry], _docs(), min_entries=1)
    assert not report.ok


def test_type_four_requires_a_present_neighbour() -> None:
    entry = _entry(
        type=4,
        absent_patterns=(r"source code escrow",),
        present_patterns=(r"escrow",),
    )
    assert validate_negatives([entry], _docs(), min_entries=1).ok

    missing = _entry(
        type=4,
        absent_patterns=(r"source code escrow",),
        present_patterns=(r"non-compete",),
    )
    report = validate_negatives([missing], _docs(), min_entries=1)
    assert not report.ok
    assert "near-miss neighbour missing" in report.violations[0].problem


def test_type_four_without_patterns_is_a_violation() -> None:
    report = validate_negatives([_entry(type=4)], _docs(), min_entries=1)
    assert not report.ok
    assert "needs present_patterns" in report.violations[0].problem


def test_duplicate_ids_are_rejected() -> None:
    report = validate_negatives([_entry(), _entry()], _docs(), min_entries=1)
    assert not report.ok
    assert any("duplicate id" in v.problem for v in report.violations)


def test_invalid_type_is_rejected() -> None:
    report = validate_negatives([_entry(type=9)], _docs(), min_entries=1)
    assert not report.ok


def test_unknown_scope_is_rejected() -> None:
    report = validate_negatives([_entry(scope="cuad/nope.txt")], _docs(), min_entries=1)
    assert not report.ok
    assert "unknown scope" in report.violations[0].problem


def test_missing_absent_patterns_is_rejected() -> None:
    report = validate_negatives([_entry(absent_patterns=())], _docs(), min_entries=1)
    assert not report.ok


def test_minimum_entry_count_is_enforced() -> None:
    report = validate_negatives([_entry()], _docs(), min_entries=40)
    assert not report.ok
    assert any("minimum" in v.problem for v in report.violations)


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(tmp_path: Path) -> None:
    entries = [_entry(id="A"), _entry(id="B", type=4, present_patterns=(r"escrow",))]
    path = tmp_path / "negatives.jsonl"
    save_negatives(entries, path)

    restored = load_negatives(path)
    assert [e.id for e in restored] == ["A", "B"]
    assert restored[1].present_patterns == (r"escrow",)


def test_split_is_derived_not_stored() -> None:
    entry = _entry(query="Some query text that hashes somewhere")
    assert entry.split == assign_split(entry.query)


def test_load_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "X"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing field"):
        load_negatives(path)


# --------------------------------------------------------------------------- #
# The frozen file
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SET_AVAILABLE, reason="negative set not created yet")
def test_frozen_set_is_valid() -> None:
    entries = load_negatives(NEGATIVES_PATH)
    _, documents = load_mini_cuad()
    report = validate_negatives(entries, documents)

    assert report.ok, [str(v) for v in report.violations]
    assert len(entries) >= 40
    assert set(report.per_type) == {1, 2, 3, 4}
    assert report.per_split["train"] > 0 and report.per_split["test"] > 0


@pytest.mark.skipif(not SET_AVAILABLE or not DATA_AVAILABLE, reason="needs data")
def test_frozen_set_does_not_duplicate_benchmark_queries() -> None:
    entries = load_negatives(NEGATIVES_PATH)
    benchmark = {query.text.strip() for query in load_benchmark("cuad")}
    assert not [entry.id for entry in entries if entry.query.strip() in benchmark]


@pytest.mark.skipif(not SET_AVAILABLE, reason="negative set not created yet")
def test_frozen_set_ids_are_sequential() -> None:
    entries = load_negatives(NEGATIVES_PATH)
    assert [entry.id for entry in entries] == [f"N{i:02d}" for i in range(1, len(entries) + 1)]
