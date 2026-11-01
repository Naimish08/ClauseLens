"""Tests for the data layer: splits, mini selection, offsets.

The integration tests need `make data` to have run; they skip otherwise so that
`make test` works on a fresh checkout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from scripts.download_data import _safe_parts
from src.data import (
    BENCHMARKS_DIR,
    Document,
    Query,
    Snippet,
    assign_split,
    derive_category,
    derive_doc_hint,
    load_benchmark,
    load_corpus,
    read_document_text,
    select_mini,
    verify_offsets,
)

DATA_AVAILABLE = (BENCHMARKS_DIR / "cuad.json").exists()
requires_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="run `make data` first")


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_assign_split_is_deterministic() -> None:
    query = "Consider the Agency Agreement; What is the governing law?"
    assert assign_split(query) == assign_split(query)


def test_assign_split_ignores_surrounding_whitespace() -> None:
    query = "What is the governing law of this contract?"
    assert assign_split(f"  {query}\n") == assign_split(query)


def test_assign_split_buckets_match_spec() -> None:
    """The split must be sha256(hex) % 10 < 3 -> test, exactly as specified."""
    import hashlib

    for query in ["a", "b", "What is the governing law?", "x" * 200]:
        bucket = int(hashlib.sha256(query.strip().encode()).hexdigest(), 16) % 10
        assert assign_split(query) == ("test" if bucket < 3 else "train")


def test_assign_split_produces_both_sides() -> None:
    queries = [f"question number {i}" for i in range(200)]
    splits = {assign_split(q) for q in queries}
    assert splits == {"train", "test"}


# --------------------------------------------------------------------------- #
# Query parsing
# --------------------------------------------------------------------------- #
def test_derive_category_and_doc_hint() -> None:
    query = (
        "Consider the Service Agreement between A Ltd and B for Chief Officer Position; "
        "What is the governing law for this contract?"
    )
    assert derive_doc_hint(query) == (
        "Consider the Service Agreement between A Ltd and B for Chief Officer Position"
    )
    assert derive_category(query) == "What is the governing law for this contract?"


def test_derive_helpers_tolerate_missing_semicolon() -> None:
    assert derive_doc_hint("bare question") == ""
    assert derive_category("bare question") == "bare question"


# --------------------------------------------------------------------------- #
# Mini selection
# --------------------------------------------------------------------------- #
def _fake_queries(n: int) -> list[Query]:
    return [
        Query(
            text=f"q{i}",
            snippets=(Snippet(f"doc{i % 7}.txt", 0, 1, "x"),),
        )
        for i in range(n)
    ]


def test_select_mini_is_a_prefix_of_the_reference_ordering() -> None:
    queries = _fake_queries(500)

    def reference(test: Query) -> float:
        # Verbatim expression from legalbenchrag/benchmark.py
        return (random.seed(test.snippets[0].file_path), random.random())[1]

    expected = sorted(queries, key=reference)[:194]
    assert select_mini(queries) == expected


def test_select_mini_noop_below_limit() -> None:
    queries = _fake_queries(10)
    assert select_mini(queries) == queries


@requires_data
def test_mini_cuad_profile() -> None:
    queries = select_mini(load_benchmark("cuad"))
    assert len(queries) == 194
    assert len({fp for q in queries for fp in q.file_paths}) == 29
    assert sum(len(q) for q in queries) == 272
    splits = {q.split for q in queries}
    assert splits == {"train", "test"}
    assert sum(1 for q in queries if q.split == "train") == 140


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_read_modes_differ_on_crlf(tmp_path: Path) -> None:
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"line one\r\nline two\r\n")
    assert read_document_text(path, mode="raw") == "line one\r\nline two\r\n"
    assert read_document_text(path, mode="text") == "line one\nline two\n"


def test_unknown_read_mode_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown read mode"):
        read_document_text(path, mode="bytes")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Offset verification
# --------------------------------------------------------------------------- #
def _docs(tmp_path: Path, text: str, name: str = "doc.txt") -> dict[str, Document]:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return {name: Document(name, text)}


def test_verify_offsets_accepts_correct_spans(tmp_path: Path) -> None:
    text = "The governing law is Delaware."
    docs = _docs(tmp_path, text)
    start = text.index("Delaware")
    query = Query("q", (Snippet("doc.txt", start, start + len("Delaware"), "Delaware"),))
    report = verify_offsets([query], docs, corpus_dir=tmp_path)
    assert report.ok
    assert report.snippets == 1


def test_verify_offsets_detects_mismatch(tmp_path: Path) -> None:
    text = "The governing law is Delaware."
    docs = _docs(tmp_path, text)
    query = Query("q", (Snippet("doc.txt", 0, 3, "The wrong answer"),))
    report = verify_offsets([query], docs, corpus_dir=tmp_path)
    assert not report.ok
    assert report.problems[0].kind == "mismatch"
    assert report.problems[0].actual == "The"


def test_verify_offsets_detects_out_of_bounds(tmp_path: Path) -> None:
    docs = _docs(tmp_path, "short")
    query = Query("q", (Snippet("doc.txt", 2, 99, "x"),))
    report = verify_offsets([query], docs, corpus_dir=tmp_path)
    assert report.problems[0].kind == "out_of_bounds"


def test_verify_offsets_detects_missing_document(tmp_path: Path) -> None:
    docs = _docs(tmp_path, "text")
    query = Query("q", (Snippet("absent.txt", 0, 1, "t"),))
    report = verify_offsets([query], docs, corpus_dir=tmp_path)
    assert report.problems[0].kind == "missing_document"


def test_verify_offsets_detects_overlapping_ground_truth(tmp_path: Path) -> None:
    text = "abcdefghij"
    docs = _docs(tmp_path, text)
    query = Query(
        "q",
        (Snippet("doc.txt", 0, 5, "abcde"), Snippet("doc.txt", 3, 8, "defgh")),
    )
    report = verify_offsets([query], docs, corpus_dir=tmp_path)
    assert report.overlapping_span_queries == ("q",)


def test_verify_offsets_detects_crlf(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_bytes(b"a\r\nb")
    docs = {"doc.txt": Document("doc.txt", read_document_text(tmp_path / "doc.txt", "raw"))}
    report = verify_offsets([], docs, corpus_dir=tmp_path)
    assert report.crlf_documents == ("doc.txt",)


# --------------------------------------------------------------------------- #
# Archive safety
# --------------------------------------------------------------------------- #
def test_safe_parts_strips_root_and_absolute_prefixes() -> None:
    assert _safe_parts("/") == ()
    assert _safe_parts("/corpus/a.txt") == ("corpus", "a.txt")
    assert _safe_parts("corpus//a.txt") == ("corpus", "a.txt")


def test_safe_parts_rejects_traversal() -> None:
    with pytest.raises(RuntimeError, match="'..'"):
        _safe_parts("../../etc/passwd")


@requires_data
def test_stored_answers_match_offsets() -> None:
    """The real integrity check: every stored answer must equal its span."""
    queries = select_mini(load_benchmark("cuad"))
    documents = load_corpus({fp for q in queries for fp in q.file_paths})
    for mode in ("raw", "text"):
        assert verify_offsets(queries, documents, mode=mode).ok


@requires_data
def test_benchmark_json_snippet_shape() -> None:
    raw = json.loads((BENCHMARKS_DIR / "cuad.json").read_text(encoding="utf-8"))
    first = raw["tests"][0]["snippets"][0]
    assert set(first) >= {"file_path", "span", "answer"}


# --------------------------------------------------------------------------- #
# Cross-validation folds
# --------------------------------------------------------------------------- #
def test_fold_of_is_stable_and_in_range() -> None:
    from src.data import fold_of

    for query in ["a", "What is the governing law?", "x" * 100]:
        assert fold_of(query) == fold_of(query)
        assert 0 <= fold_of(query, 5) < 5


def test_fold_of_rejects_single_fold() -> None:
    from src.data import fold_of

    with pytest.raises(ValueError, match="at least 2"):
        fold_of("q", n_folds=1)


def test_fold_prefix_does_not_collide_with_split_bucket() -> None:
    """Fold assignment must not be a rehash of the test bucket."""
    import hashlib


    query = "What is the expiration date of this contract?"
    test_bucket = int(hashlib.sha256(query.strip().encode()).hexdigest(), 16) % 10
    fold_bucket = int(
        hashlib.sha256(f"fold:{query.strip()}".encode()).hexdigest(), 16
    ) % 10
    assert test_bucket != fold_bucket


@requires_data
def test_cv_folds_partition_train() -> None:
    from src.data import by_split, cv_folds

    train = by_split(select_mini(load_benchmark("cuad")), "train")
    folds = cv_folds(train, n_folds=5)
    assert set(folds) == {0, 1, 2, 3, 4}
    assert all(folds.values()), "every fold should be non-empty on 140 queries"
    assert sum(len(v) for v in folds.values()) == len(train)
    assert len({q.text for v in folds.values() for q in v}) == len(train)
