"""Tests for dense, hybrid and rerank retrieval using fake encoders.

No model download and no torch: the encoder is a protocol, so the retrieval logic
is tested independently of the weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.retrieval.base import tokenize
from src.retrieval.dense import (
    MODEL_SPECS,
    DenseRetriever,
    prepare_passages,
    prepare_query,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import RerankRetriever
from src.types import make_chunk

VOCAB = ("governing", "law", "royalty", "audit", "insurance")


def chunks(*texts: str, file_path: str = "cuad/a.txt"):
    out = []
    offset = 0
    for text in texts:
        out.append(make_chunk(file_path, offset, offset + len(text), text))
        offset += len(text) + 1
    return out


class BagOfWordsEncoder:
    """Deterministic stand-in for a sentence encoder."""

    name = "fake"
    dim = len(VOCAB)

    def __init__(self) -> None:
        self.passage_calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            if token in VOCAB:
                vector[VOCAB.index(token)] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def encode_passages(self, texts):
        self.passage_calls += 1
        return np.vstack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class ExplodingEncoder(BagOfWordsEncoder):
    def encode_passages(self, texts):
        raise AssertionError("cache should have prevented re-encoding")


class FixedRetriever:
    """Returns a predetermined ranking, for RRF arithmetic."""

    name = "fixed"

    def __init__(self, ranking: list[str], chunks_by_id: dict) -> None:
        self.ranking = ranking
        self.chunks_by_id = chunks_by_id
        self.requested_k = None

    def index(self, chunks):
        pass

    def search(self, query, k=10):
        self.requested_k = k
        out = []
        for rank, chunk_id in enumerate(self.ranking):
            chunk = self.chunks_by_id[chunk_id]
            out.append(
                type("S", (), {"file_path": chunk.file_path, "start": chunk.start,
                               "end": chunk.end, "score": 1.0 / (rank + 1),
                               "chunk_id": chunk.chunk_id})()
            )
        return out[:k]


# --------------------------------------------------------------------------- #
# Prefixes
# --------------------------------------------------------------------------- #
def test_e5_prefixes_both_sides() -> None:
    spec = MODEL_SPECS["e5-large-v2"]
    assert prepare_query(spec, "q") == "query: q"
    assert prepare_passages(spec, ["p"]) == ["passage: p"]


def test_bge_prefixes_only_the_query() -> None:
    spec = MODEL_SPECS["bge-base-en-v1.5"]
    assert prepare_query(spec, "q").startswith("Represent this sentence")
    assert prepare_passages(spec, ["p"]) == ["p"]


def test_gte_adds_no_prefix() -> None:
    spec = MODEL_SPECS["gte-base"]
    assert prepare_query(spec, "q") == "q"
    assert prepare_passages(spec, ["p"]) == ["p"]


# --------------------------------------------------------------------------- #
# Dense retrieval
# --------------------------------------------------------------------------- #
def test_dense_ranks_the_matching_chunk_first() -> None:
    corpus = chunks(
        "the governing law of this agreement",
        "a royalty is payable on net sales",
        "the party shall maintain insurance",
    )
    retriever = DenseRetriever(encoder=BagOfWordsEncoder())
    retriever.index(corpus)
    results = retriever.search("governing law", k=3)
    assert results
    assert results[0].start == corpus[0].start
    assert results[0].chunk_id == corpus[0].chunk_id


def test_dense_returns_indexed_spans_only() -> None:
    corpus = chunks("governing law", "royalty", "insurance")
    retriever = DenseRetriever(encoder=BagOfWordsEncoder())
    retriever.index(corpus)
    indexed = {chunk.chunk_id for chunk in corpus}
    for result in retriever.search("royalty", k=3):
        assert result.chunk_id in indexed


def test_dense_on_empty_index_returns_nothing() -> None:
    retriever = DenseRetriever(encoder=BagOfWordsEncoder())
    retriever.index([])
    assert retriever.search("anything") == []


def test_dense_ignores_non_positive_similarity() -> None:
    corpus = chunks("governing law")
    retriever = DenseRetriever(encoder=BagOfWordsEncoder())
    retriever.index(corpus)
    assert retriever.search("royalty insurance", k=3) == []


def test_dense_caches_embeddings(tmp_path: Path) -> None:
    corpus = chunks("governing law", "royalty", "insurance")
    first = DenseRetriever(encoder=BagOfWordsEncoder(), cache_dir=tmp_path)
    first.index(corpus)
    assert first.cache_hit is False
    assert len(list(tmp_path.glob("*.npz"))) == 1

    second = DenseRetriever(encoder=ExplodingEncoder(), cache_dir=tmp_path)
    second.index(corpus)
    assert second.cache_hit is True
    assert second.search("royalty", k=1)[0].chunk_id == corpus[1].chunk_id


def test_dense_cache_key_depends_on_chunks(tmp_path: Path) -> None:
    first = DenseRetriever(encoder=BagOfWordsEncoder(), cache_dir=tmp_path)
    first.index(chunks("governing law"))
    second = DenseRetriever(encoder=BagOfWordsEncoder(), cache_dir=tmp_path)
    second.index(chunks("royalty"))
    assert second.cache_hit is False


# --------------------------------------------------------------------------- #
# Hybrid RRF
# --------------------------------------------------------------------------- #
def test_hybrid_fuses_by_reciprocal_rank() -> None:
    corpus = chunks("alpha", "beta", "gamma")
    by_id = {chunk.chunk_id: chunk for chunk in corpus}
    left = FixedRetriever([corpus[0].chunk_id, corpus[1].chunk_id], by_id)
    right = FixedRetriever([corpus[1].chunk_id, corpus[2].chunk_id], by_id)

    hybrid = HybridRetriever(members=[left, right], k_rrf=60, depth=10)
    hybrid.index(corpus)
    results = hybrid.search("q", k=3)

    expected = {
        corpus[0].chunk_id: 1 / 61,
        corpus[1].chunk_id: 1 / 62 + 1 / 61,
        corpus[2].chunk_id: 1 / 62,
    }
    assert results[0].chunk_id == corpus[1].chunk_id
    for result in results:
        assert result.score == pytest.approx(expected[result.chunk_id])


def test_hybrid_weights_change_the_winner() -> None:
    corpus = chunks("alpha", "beta")
    by_id = {chunk.chunk_id: chunk for chunk in corpus}
    left = FixedRetriever([corpus[0].chunk_id], by_id)
    right = FixedRetriever([corpus[1].chunk_id], by_id)

    hybrid = HybridRetriever(members=[left, right], weights=[1.0, 5.0])
    hybrid.index(corpus)
    assert hybrid.search("q", k=1)[0].chunk_id == corpus[1].chunk_id


def test_hybrid_passes_depth_to_members() -> None:
    corpus = chunks("alpha")
    by_id = {chunk.chunk_id: chunk for chunk in corpus}
    member = FixedRetriever([corpus[0].chunk_id], by_id)
    hybrid = HybridRetriever(members=[member], depth=42)
    hybrid.index(corpus)
    hybrid.search("q", k=1)
    assert member.requested_k == 42


def test_hybrid_requires_members() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        HybridRetriever(members=[])


def test_hybrid_rejects_misaligned_weights() -> None:
    corpus = chunks("alpha")
    by_id = {chunk.chunk_id: chunk for chunk in corpus}
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(members=[FixedRetriever([], by_id)], weights=[1.0, 2.0])


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
class KeywordCrossEncoder:
    name = "fake-cross"

    def score(self, query: str, passages):
        wanted = set(tokenize(query))
        return [float(len(wanted & set(tokenize(passage)))) for passage in passages]


def test_rerank_reorders_by_cross_encoder_score() -> None:
    corpus = chunks(
        "insurance obligations apply",
        "the governing law is delaware",
        "royalty payments",
    )
    base = FixedRetriever([corpus[0].chunk_id, corpus[1].chunk_id, corpus[2].chunk_id],
                          {chunk.chunk_id: chunk for chunk in corpus})
    reranker = RerankRetriever(base=base, candidates=3, encoder=KeywordCrossEncoder())
    reranker.index(corpus)

    results = reranker.search("governing law", k=3)
    assert results[0].chunk_id == corpus[1].chunk_id


def test_rerank_respects_k() -> None:
    corpus = chunks("a", "b", "c")
    base = FixedRetriever([chunk.chunk_id for chunk in corpus], {c.chunk_id: c for c in corpus})
    reranker = RerankRetriever(base=base, candidates=3, encoder=KeywordCrossEncoder())
    reranker.index(corpus)
    assert len(reranker.search("a", k=2)) == 2


def test_rerank_requests_the_candidate_pool() -> None:
    corpus = chunks("a", "b")
    base = FixedRetriever([c.chunk_id for c in corpus], {c.chunk_id: c for c in corpus})
    reranker = RerankRetriever(base=base, candidates=50, encoder=KeywordCrossEncoder())
    reranker.index(corpus)
    reranker.search("a", k=2)
    assert base.requested_k == 50


def test_rerank_handles_an_empty_pool() -> None:
    corpus = chunks("a")
    base = FixedRetriever([], {})
    reranker = RerankRetriever(base=base, candidates=5, encoder=KeywordCrossEncoder())
    reranker.index(corpus)
    assert reranker.search("a", k=5) == []


def test_rerank_rejects_bad_candidate_count() -> None:
    with pytest.raises(ValueError, match="candidates"):
        RerankRetriever(base=FixedRetriever([], {}), candidates=0)
