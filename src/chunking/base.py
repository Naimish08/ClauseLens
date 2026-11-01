"""Chunker protocol and span-contract validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.data import Document
from src.types import Chunk


class Chunker(Protocol):
    """Turns one document into span-exact chunks."""

    name: str

    def chunk(self, document: Document) -> list[Chunk]: ...


def validate_chunks(document: Document, chunks: Sequence[Chunk]) -> None:
    """Assert the span contract (I3) for every chunk of one document.

    Cheap enough to run on every index build, and it turns a whole class of silent
    metric corruption into an immediate, local error.
    """
    for chunk in chunks:
        if chunk.file_path != document.file_path:
            raise ValueError(
                f"chunk {chunk.chunk_id} belongs to {chunk.file_path}, not {document.file_path}"
            )
        if not 0 <= chunk.start < chunk.end <= len(document.content):
            raise ValueError(f"chunk span out of bounds: {chunk.chunk_id}")
        actual = document.content[chunk.start : chunk.end]
        if actual != chunk.text:
            raise ValueError(
                f"span contract violated for {chunk.chunk_id}: "
                f"expected {actual[:40]!r}, got {chunk.text[:40]!r}"
            )
