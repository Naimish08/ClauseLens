"""Hybrid retrieval with Reciprocal Rank Fusion.

RRF is used instead of score normalisation because BM25 scores and cosine
similarities live on incomparable scales, and normalising them would add a
fragile choice on top of an already-noisy comparison.

    score(d) = Σ_r  w_r / (k_rrf + rank_r(d))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.retrieval.base import Retriever
from src.types import Chunk, RetrievedSpan, sort_retrieved


@dataclass
class HybridRetriever:
    """Fuses several retrievers by rank, not by score."""

    members: list[Retriever]
    weights: list[float] | None = None
    k_rrf: int = 60
    depth: int = 100
    name: str = "hybrid"

    _chunks: list[Chunk] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("hybrid retrieval needs at least one member")
        if self.weights is None:
            self.weights = [1.0] * len(self.members)
        if len(self.weights) != len(self.members):
            raise ValueError("weights must align with members")
        if self.k_rrf <= 0:
            raise ValueError("k_rrf must be positive")

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        for member in self.members:
            member.index(chunks)

    def search(self, query: str, k: int = 10) -> list[RetrievedSpan]:
        fused: dict[str, float] = {}
        by_id: dict[str, RetrievedSpan] = {}

        for member, weight in zip(self.members, self.weights or [], strict=True):
            for rank, span in enumerate(member.search(query, k=self.depth), start=1):
                key = span.chunk_id or f"{span.file_path}:{span.start}:{span.end}"
                fused[key] = fused.get(key, 0.0) + weight / (self.k_rrf + rank)
                by_id.setdefault(key, span)

        ranked = [
            RetrievedSpan(
                file_path=by_id[key].file_path,
                start=by_id[key].start,
                end=by_id[key].end,
                score=score,
                chunk_id=by_id[key].chunk_id,
            )
            for key, score in fused.items()
            if score > 0.0
        ]
        return sort_retrieved(ranked)[:k]

    def __len__(self) -> int:
        return len(self._chunks)
