"""Retriever/reranker registries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.retrieval.base import Reranker, Retriever, tokenize
from src.retrieval.bm25 import BM25Retriever

RetrieverFactory = Callable[..., Retriever]

RETRIEVERS: dict[str, RetrieverFactory] = {
    "bm25": BM25Retriever,
}

#: Rerankers are optional; ``none`` means the retrieval ranking stands.
RERANKERS: dict[str, Callable[..., Reranker] | None] = {
    "none": None,
}


def build_retriever(name: str, params: dict[str, Any] | None = None) -> Retriever:
    if name not in RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}; available: {sorted(RETRIEVERS)}")
    return RETRIEVERS[name](**(params or {}))


def build_reranker(name: str, params: dict[str, Any] | None = None) -> Reranker | None:
    if name not in RERANKERS:
        raise ValueError(f"unknown reranker {name!r}; available: {sorted(RERANKERS)}")
    factory = RERANKERS[name]
    return None if factory is None else factory(**(params or {}))


__all__ = [
    "Reranker",
    "Retriever",
    "RETRIEVERS",
    "RERANKERS",
    "BM25Retriever",
    "build_retriever",
    "build_reranker",
    "tokenize",
]
