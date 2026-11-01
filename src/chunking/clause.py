"""Clause-aware chunking: one chunk per detected section.

Contract formatting is adversarial — inconsistent numbering, all-caps headings,
tables, page headers mid-sentence — so this is a small hand-written heading
detector rather than a library. It is a heuristic and is measured, not assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.chunking.base import apply_overlap
from src.chunking.recursive import merge_spans, split_span
from src.data import Document
from src.types import Chunk, make_chunk

#: Numbered or titled headings. Each pattern is matched per line.
HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[ \t]*(?:SECTION|ARTICLE)\s+[0-9IVXLC]+\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[ \t]*\d+(?:\.\d+)+\.?[ \t]+\S", re.MULTILINE),
    re.compile(r"^[ \t]*\d+\.[ \t]+\S", re.MULTILINE),
    re.compile(r"^[ \t]*[A-Z][A-Z0-9 ,.&/'()\-]{3,80}[ \t]*$", re.MULTILINE),
)


def heading_offsets(text: str) -> list[int]:
    """Start offsets of likely clause headings, sorted and de-duplicated."""
    offsets: set[int] = set()
    for pattern in HEADING_PATTERNS:
        for match in pattern.finditer(text):
            offsets.add(match.start())
    return sorted(offset for offset in offsets if 0 < offset < len(text))


def _coalesce(
    spans: list[tuple[int, int]], min_size: int, max_size: int
) -> list[tuple[int, int]]:
    """Grow undersized sections into their neighbour, never past ``max_size``."""
    if not spans:
        return []

    merged: list[tuple[int, int]] = []
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        undersized = current_end - current_start < min_size
        if undersized and end - current_start <= max_size:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    # A trailing fragment that is still too small joins the previous chunk if it fits.
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < min_size:
        combined = (merged[-2][0], merged[-1][1])
        if combined[1] - combined[0] <= max_size:
            merged[-2:] = [combined]
    return merged


@dataclass(frozen=True, slots=True)
class ClauseChunker:
    """One chunk per numbered section or heading, split only if oversized."""

    max_size: int = 1024
    min_size: int = 200
    overlap: int = 0
    name: str = "clause"

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.min_size < 0:
            raise ValueError("min_size must be non-negative")
        if not 0 <= self.overlap < self.max_size:
            raise ValueError("overlap must be in [0, max_size)")

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []

        boundaries = [0, *heading_offsets(text), len(text)]
        sections = [(a, b) for a, b in zip(boundaries, boundaries[1:], strict=False) if b > a]

        spans: list[tuple[int, int]] = []
        for start, end in sections:
            if end - start <= self.max_size:
                spans.append((start, end))
            else:
                pieces = merge_spans(
                    split_span(text, start, end, self.max_size), self.max_size
                )
                spans.extend(pieces)

        spans = _coalesce(spans, self.min_size, self.max_size)
        spans = apply_overlap(spans, self.overlap)
        return [make_chunk(document.file_path, s, e, text[s:e]) for s, e in spans]
