"""Retriever and reranker protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.types import Chunk, RetrievedSpan


class Retriever(Protocol):
    """Indexes chunks once, then answers queries."""

    name: str

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int = 10) -> list[RetrievedSpan]: ...


class Reranker(Protocol):
    """Reorders candidate spans; never queries the index itself."""

    name: str

    def rerank(
        self, query: str, candidates: Sequence[RetrievedSpan], k: int = 10
    ) -> list[RetrievedSpan]: ...


def tokenize(text: str) -> list[str]:
    """Shared tokenisation: lowercase alphanumerics, single characters dropped.

    Stop-words are deliberately kept — legal text is dense in "shall",
    "notwithstanding" and "hereof", and removing them measurably weakens BM25.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            if len(current) > 1:
                tokens.append("".join(current))
            current.clear()
    if len(current) > 1:
        tokens.append("".join(current))
    return tokens
