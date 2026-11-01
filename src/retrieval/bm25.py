"""Okapi BM25, written in-house (ADR-007).

Why not a library: scores must map back to exact chunk spans, the tokeniser is
tuned for legal text, and the whole thing needs to be explainable line by line.
It is also the baseline that dense retrieval has to beat to justify its cost.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.retrieval.base import tokenize
from src.types import Chunk, RetrievedSpan, sort_retrieved


@dataclass
class BM25Retriever:
    """Standard Okapi BM25 over chunk text.

    ``k1`` controls term-frequency saturation, ``b`` the length normalisation.
    """

    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"

    _chunks: list[Chunk] = field(default_factory=list, init=False, repr=False)
    _postings: dict[str, list[tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _doc_len: list[int] = field(default_factory=list, init=False, repr=False)
    _avgdl: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError("b must be in [0, 1]")

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._doc_len = []
        postings: dict[str, list[tuple[int, int]]] = {}

        for doc_index, chunk in enumerate(self._chunks):
            tokens = tokenize(chunk.text)
            self._doc_len.append(len(tokens))
            for term, frequency in Counter(tokens).items():
                postings.setdefault(term, []).append((doc_index, frequency))

        self._postings = postings
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

    def idf(self, term: str) -> float:
        """Standard BM25 IDF: rare terms score higher, common terms ~0."""
        document_frequency = len(self._postings.get(term, ()))
        if document_frequency == 0:
            return 0.0
        n_docs = len(self._chunks)
        return math.log((n_docs - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0)

    def search(self, query: str, k: int = 10) -> list[RetrievedSpan]:
        if not self._chunks or k <= 0:
            return []

        scores: dict[int, float] = {}
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            if idf <= 0.0:
                continue
            for doc_index, frequency in postings:
                norm = 1.0 - self.b + self.b * (self._doc_len[doc_index] / self._avgdl)
                contribution = idf * (frequency * (self.k1 + 1.0)) / (
                    frequency + self.k1 * norm
                )
                scores[doc_index] = scores.get(doc_index, 0.0) + contribution

        if not scores:
            return []

        ranked = [
            RetrievedSpan(
                file_path=chunk.file_path,
                start=chunk.start,
                end=chunk.end,
                score=score,
                chunk_id=chunk.chunk_id,
            )
            for doc_index, score in scores.items()
            if score > 0.0
            for chunk in (self._chunks[doc_index],)
        ]
        return sort_retrieved(ranked)[:k]

    def __len__(self) -> int:
        return len(self._chunks)
