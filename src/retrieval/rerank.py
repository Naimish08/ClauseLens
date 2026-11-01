"""Cross-encoder reranking as a retriever wrapper.

Reranking is composed rather than bolted on: it wraps a base retriever, fetches a
candidate list, and reorders it. That keeps the pipeline's interface unchanged and
makes "the reranker adds N milliseconds" a measurable statement.

The reranker needs chunk text, which ``RetrievedSpan`` deliberately does not carry,
so ``index`` stores a text lookup. Nothing else about retrieval changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from src.retrieval.base import Retriever
from src.types import Chunk, RetrievedSpan, sort_retrieved


class CrossEncoder(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class SentenceTransformerCrossEncoder:
    """Lazy wrapper over a sentence-transformers CrossEncoder."""

    def __init__(self, model: str = "BAAI/bge-reranker-base", *, batch_size: int = 32) -> None:
        self.model = model
        self.batch_size = batch_size
        self.name = model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        model = self._load()
        pairs = [(query, passage) for passage in passages]
        if not pairs:
            return []
        return [float(value) for value in model.predict(pairs, batch_size=self.batch_size)]


@dataclass
class RerankRetriever:
    """Retrieve ``candidates``, then reorder with a cross-encoder."""

    base: Retriever
    candidates: int = 50
    model: str = "BAAI/bge-reranker-base"
    name: str = "rerank"
    encoder: CrossEncoder | None = None

    _text_by_id: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    last_rerank_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")
        if self.encoder is None:
            self.encoder = SentenceTransformerCrossEncoder(self.model)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.base.index(chunks)
        self._text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}

    def search(self, query: str, k: int = 10) -> list[RetrievedSpan]:
        import time

        pool = self.base.search(query, k=max(self.candidates, k))
        if not pool:
            self.last_rerank_ms = 0.0
            return []

        passages = [self._text_by_id.get(span.chunk_id, "") for span in pool]
        started = time.perf_counter()
        assert self.encoder is not None
        scores = self.encoder.score(query, passages)
        self.last_rerank_ms = (time.perf_counter() - started) * 1000.0

        reranked = [
            RetrievedSpan(
                file_path=span.file_path,
                start=span.start,
                end=span.end,
                score=float(score),
                chunk_id=span.chunk_id,
            )
            for span, score in zip(pool, scores, strict=True)
        ]
        return sort_retrieved(reranked)[:k]

    def __len__(self) -> int:
        return len(self._text_by_id)
