"""Recursive character splitting, with exact character spans.

Splits on the coarsest separator that works (paragraph, line, sentence, word),
falling back to a hard cut. Unlike a text-based splitter, this operates on
`(start, end)` intervals so every chunk still satisfies the span contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chunking.base import apply_overlap
from src.data import Document
from src.types import Chunk, make_chunk

SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def split_span(
    text: str,
    start: int,
    end: int,
    size: int,
    separators: tuple[str, ...] = SEPARATORS,
) -> list[tuple[int, int]]:
    """Split one interval into pieces no longer than ``size`` characters."""
    if end - start <= size:
        return [(start, end)]

    for index, separator in enumerate(separators):
        if separator == "":
            return [(cursor, min(cursor + size, end)) for cursor in range(start, end, size)]
        if text.find(separator, start, end) == -1:
            continue

        pieces: list[tuple[int, int]] = []
        cursor = start
        while cursor < end:
            found = text.find(separator, cursor, end)
            if found == -1:
                pieces.append((cursor, end))
                break
            cut = found + len(separator)
            pieces.append((cursor, cut))
            cursor = cut

        out: list[tuple[int, int]] = []
        for piece_start, piece_end in pieces:
            if piece_end - piece_start <= size:
                out.append((piece_start, piece_end))
            else:
                out.extend(split_span(text, piece_start, piece_end, size, separators[index + 1 :]))
        return out

    return [(start, end)]


def merge_spans(spans: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Greedily pack contiguous spans into windows no longer than ``size``."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and end - merged[-1][0] <= size:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    """Separator-hierarchy chunker, the usual "smart default" baseline."""

    size: int = 512
    overlap: int = 0
    name: str = "recursive"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be in [0, size)")

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []
        pieces = merge_spans(split_span(text, 0, len(text), self.size), self.size)
        spans = apply_overlap(pieces, self.overlap)
        return [make_chunk(document.file_path, s, e, text[s:e]) for s, e in spans]
