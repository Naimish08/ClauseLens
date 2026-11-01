"""Tests for BM25 and tokenisation."""

from __future__ import annotations

import math

import pytest

from src.retrieval import build_retriever
from src.retrieval.base import tokenize
from src.retrieval.bm25 import BM25Retriever
from src.types import make_chunk


def chunks(*texts: str, file_path: str = "cuad/a.txt"):
    out = []
    offset = 0
    for text in texts:
        out.append(make_chunk(file_path, offset, offset + len(text), text))
        offset += len(text) + 1
    return out


@pytest.fixture
def corpus():
    return chunks(
        "the governing law of this agreement is delaware",
        "each party shall maintain the confidentiality of information",
        "the agreement may be terminated for cause with written notice",
    )


def test_tokenize_lowercases_and_keeps_stopwords() -> None:
    assert tokenize("The GOVERNING law, of this Agreement!") == [
        "the",
        "governing",
        "law",
        "of",
        "this",
        "agreement",
    ]


def test_tokenize_drops_single_characters() -> None:
    assert tokenize("a b c word") == ["word"]


def test_bm25_ranks_the_matching_chunk_first(corpus) -> None:
    retriever = BM25Retriever()
    retriever.index(corpus)
    results = retriever.search("governing law", k=3)
    assert results
    assert results[0].file_path == "cuad/a.txt"
    assert corpus[0].start == results[0].start


def test_bm25_returns_indexed_spans_only(corpus) -> None:
    retriever = BM25Retriever()
    retriever.index(corpus)
    indexed = {chunk.chunk_id for chunk in corpus}
    for result in retriever.search("terminated written notice", k=3):
        assert result.chunk_id in indexed


def test_bm25_scores_match_the_formula(corpus) -> None:
    retriever = BM25Retriever(k1=1.2, b=0.5)
    retriever.index(corpus)
    k1, b = 1.2, 0.5

    lengths = [len(tokenize(chunk.text)) for chunk in corpus]
    avgdl = sum(lengths) / len(lengths)
    n_docs = len(corpus)

    expected = 0.0
    for term in tokenize("agreement delaware"):
        df = sum(1 for chunk in corpus if term in tokenize(chunk.text))
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf = tokenize(corpus[0].text).count(term)
        if tf:
            expected += idf * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * lengths[0] / avgdl)
            )

    results = retriever.search("agreement delaware", k=3)
    top = next(r for r in results if r.start == corpus[0].start)
    assert top.score == pytest.approx(expected)


def test_bm25_ignores_terms_absent_from_the_index(corpus) -> None:
    retriever = BM25Retriever()
    retriever.index(corpus)
    assert retriever.search("zzzz nonexistent", k=5) == []


def test_bm25_on_empty_index_returns_nothing() -> None:
    retriever = BM25Retriever()
    retriever.index([])
    assert retriever.search("anything") == []
    assert len(retriever) == 0


def test_bm25_k_is_respected(corpus) -> None:
    retriever = BM25Retriever()
    retriever.index(corpus)
    assert len(retriever.search("agreement party", k=2)) <= 2


def test_bm25_ties_are_broken_deterministically() -> None:
    tied = [
        make_chunk("cuad/b.txt", 0, 20, "identical words here"),
        make_chunk("cuad/a.txt", 0, 20, "identical words here"),
    ]
    retriever = BM25Retriever()
    retriever.index(tied)
    # Same score, so ordering falls back to file path.
    assert [r.file_path for r in retriever.search("identical", k=2)] == [
        "cuad/a.txt",
        "cuad/b.txt",
    ]


def test_common_terms_carry_no_negative_idf(corpus) -> None:
    retriever = BM25Retriever()
    retriever.index(corpus)
    assert retriever.idf("nonexistent") == 0.0
    assert retriever.idf("agreement") > 0.0


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(k1=-1)
    with pytest.raises(ValueError):
        BM25Retriever(b=1.5)


def test_registry_builds_by_config_name() -> None:
    retriever = build_retriever("bm25", {"k1": 1.5, "b": 0.75})
    assert retriever.name == "bm25"
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever("quantum", {})
