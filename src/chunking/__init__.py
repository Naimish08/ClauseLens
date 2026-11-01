"""Chunker registry. Adding a chunker means adding one entry here."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.chunking.base import Chunker, validate_chunks
from src.chunking.fixed import FixedChunker

ChunkerFactory = Callable[..., Chunker]

CHUNKERS: dict[str, ChunkerFactory] = {
    "fixed": FixedChunker,
}


def build_chunker(name: str, params: dict[str, Any] | None = None) -> Chunker:
    if name not in CHUNKERS:
        raise ValueError(f"unknown chunker {name!r}; available: {sorted(CHUNKERS)}")
    return CHUNKERS[name](**(params or {}))


__all__ = ["Chunker", "CHUNKERS", "build_chunker", "validate_chunks", "FixedChunker"]
