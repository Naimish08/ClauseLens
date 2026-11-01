"""Sentence-window chunking: never cut a sentence in half."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.chunking.base import apply_overlap
from src.chunking.recursive import merge_spans, split_span
from src.data import Document
from src.types import Chunk, make_chunk

#: A period after one of these tokens is almost never a sentence end.
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "inc",
        "ltd",
        "co",
        "corp",
        "llc",
        "lp",
        "no",
        "p",
        "pp",
        "sec",
        "art",
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "vs",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
    }
)

_BOUNDARY = re.compile(r"(?:[.!?][\"')\]]*|\n{2,})\s*")


def _is_abbreviation(text: str, period_index: int) -> bool:
    start = period_index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in ".-"):
        start -= 1
    token = text[start:period_index].strip(".").lower()
    return token in ABBREVIATIONS


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Contiguous sentence (and paragraph) intervals covering the whole text."""
    if not text:
        return []

    boundaries = [0]
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        if end >= len(text):
            continue
        if text[match.start()] == "." and _is_abbreviation(text, match.start()):
            continue
        boundaries.append(end)
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    return [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:], strict=False)
        if end > start
    ]


@dataclass(frozen=True, slots=True)
class SentenceChunker:
    """Packs whole sentences into windows of about ``size`` characters."""

    size: int = 512
    overlap: int = 0
    max_sentences: int | None = None
    name: str = "sentence"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be in [0, size)")

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []

        windows: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end = 0
        current_count = 0

        for start, end in sentence_spans(text):
            # A single oversized sentence is split further rather than truncated.
            if end - start > self.size:
                if current_start is not None:
                    windows.append((current_start, current_end))
                    current_start, current_count = None, 0
                # Re-pack the pieces: dot-leader tables of contents can otherwise
                # explode into hundreds of near-empty chunks.
                windows.extend(merge_spans(split_span(text, start, end, self.size), self.size))
                continue

            too_long = current_start is not None and end - current_start > self.size
            too_many = (
                self.max_sentences is not None and current_count >= self.max_sentences
            )
            if current_start is None or too_long or too_many:
                if current_start is not None:
                    windows.append((current_start, current_end))
                current_start, current_end, current_count = start, end, 1
            else:
                current_end = end
                current_count += 1

        if current_start is not None:
            windows.append((current_start, current_end))

        spans = apply_overlap(windows, self.overlap)
        return [make_chunk(document.file_path, s, e, text[s:e]) for s, e in spans]
