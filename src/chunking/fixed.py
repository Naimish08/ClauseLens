"""Fixed-size chunking: the baseline against which every other chunker is judged."""

from __future__ import annotations

from dataclasses import dataclass

from src.data import Document
from src.types import Chunk, make_chunk

_BREAK_CHARS = (" ", "\n", "\t")


@dataclass(frozen=True, slots=True)
class FixedChunker:
    """Fixed-width windows with optional whitespace snapping.

    ``size`` and ``overlap`` are in characters (ADR-013). With ``snap`` enabled the
    window end is pulled back to the last whitespace within ``snap_tolerance``
    characters, so chunks rarely end mid-word; if no whitespace is near enough the
    hard cut stands.
    """

    size: int = 512
    overlap: int = 0
    snap: bool = True
    snap_tolerance: int = 80
    name: str = "fixed"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        if self.snap_tolerance < 0:
            raise ValueError("snap_tolerance must be non-negative")

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        total = len(text)
        chunks: list[Chunk] = []
        start = 0

        while start < total:
            end = min(start + self.size, total)
            if self.snap and end < total:
                end = self._snap(text, start, end)

            chunks.append(make_chunk(document.file_path, start, end, text[start:end]))

            if end >= total:
                break
            next_start = end - self.overlap
            start = next_start if next_start > start else start + 1

        return chunks

    def _snap(self, text: str, start: int, end: int) -> int:
        window_start = max(start + 1, end - self.snap_tolerance)
        best = max(text.rfind(char, window_start, end) for char in _BREAK_CHARS)
        return best + 1 if best >= window_start else end
