"""Shared record types for chunks and retrieved spans.

These live in one module so that chunkers, retrievers, metrics and the pipeline
agree on the same shape. `src/data.py` owns ground-truth records; this module owns
the things a retrieval system produces.

Invariant (I2): a `RetrievedSpan` always corresponds to exactly one indexed
`Chunk`. Retrieval never invents spans.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A span-exact slice of one document.

    Invariant (I3): ``document.content[start:end] == text``. Anything that
    normalises text must keep the normalised form out of ``text``.
    """

    chunk_id: str
    file_path: str
    start: int
    end: int
    text: str

    @property
    def span(self) -> Span:
        return (self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start

    def __len__(self) -> int:
        return self.length


@dataclass(frozen=True, slots=True)
class RetrievedSpan:
    """One search result: a chunk span plus the score that ranked it."""

    file_path: str
    start: int
    end: int
    score: float
    chunk_id: str = ""

    @property
    def span(self) -> Span:
        return (self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start

    def __len__(self) -> int:
        return self.length


def make_chunk(file_path: str, start: int, end: int, text: str) -> Chunk:
    """Build a chunk with the canonical, order-independent ID."""
    return Chunk(
        chunk_id=f"{file_path}:{start}:{end}",
        file_path=file_path,
        start=start,
        end=end,
        text=text,
    )


def sort_and_merge_spans(spans: Iterable[Span], *, max_bridge_gap_len: int = 0) -> list[Span]:
    """Sort spans and merge those that overlap or sit within ``max_bridge_gap_len``.

    Same semantics as the reference implementation's helper of the same name, so
    union-based metrics stay comparable with the paper.
    """
    ordered = sorted(spans)
    if not ordered:
        return []
    merged: list[Span] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + max_bridge_gap_len:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def rank_key(span: RetrievedSpan) -> tuple[float, str, int]:
    """Total ordering for ranking: score descending, then position ascending."""
    return (-span.score, span.file_path, span.start)


def sort_retrieved(spans: Iterable[RetrievedSpan]) -> list[RetrievedSpan]:
    """Return spans in a deterministic ranked order (I6)."""
    return sorted(spans, key=rank_key)


def merge_by_file(spans: Sequence[Span]) -> int:
    """Total length covered by ``spans`` after merging overlaps."""
    return sum(end - start for start, end in sort_and_merge_spans(spans))


def contains(outer: Span, inner: Span) -> bool:
    """True if ``inner`` lies entirely within ``outer``."""
    return outer[0] <= inner[0] and inner[1] <= outer[1]
