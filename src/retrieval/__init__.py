"""Retriever/reranker registries.

Retrievers compose: ``hybrid`` builds its members from a nested spec, and
``rerank`` wraps any base retriever (ADR-016).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.retrieval.base import Reranker, Retriever, tokenize
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, build_encoder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import RerankRetriever, SentenceTransformerCrossEncoder


def _build_bm25(params: dict[str, Any]) -> Retriever:
    return BM25Retriever(k1=params.get("k1", 1.5), b=params.get("b", 0.75))


def _build_dense(params: dict[str, Any]) -> Retriever:
    model = params.get("model", "bge-base-en-v1.5")
    return DenseRetriever(
        model=model,
        batch_size=params.get("batch_size", 64),
        cache_dir=params.get("cache_dir"),
        encoder=build_encoder(model, batch_size=params.get("batch_size", 64)),
    )


def _build_hybrid(params: dict[str, Any]) -> Retriever:
    specs = params.get("members") or []
    if not specs:
        raise ValueError("hybrid retrieval needs a non-empty members list")
    members = [build_retriever(spec["name"], spec.get("params")) for spec in specs]
    return HybridRetriever(
        members=members,
        weights=params.get("weights"),
        k_rrf=params.get("k_rrf", 60),
        depth=params.get("depth", 100),
    )


def _build_rerank(params: dict[str, Any]) -> Retriever:
    base_spec = params.get("base")
    if not base_spec:
        raise ValueError("rerank retrieval needs a base spec")
    base = build_retriever(base_spec["name"], base_spec.get("params"))
    model = params.get("model", "BAAI/bge-reranker-base")
    return RerankRetriever(
        base=base,
        candidates=params.get("candidates", 50),
        model=model,
        encoder=SentenceTransformerCrossEncoder(model),
    )


RETRIEVERS: dict[str, Callable[[dict[str, Any]], Retriever]] = {
    "bm25": _build_bm25,
    "dense": _build_dense,
    "hybrid": _build_hybrid,
    "rerank": _build_rerank,
}

#: The reranking stage is normally unused because ``rerank`` is a retriever
#: composition; kept so a future pipeline-level reranker can slot in.
RERANKERS: dict[str, Callable[..., Reranker] | None] = {
    "none": None,
}


def build_retriever(name: str, params: dict[str, Any] | None = None) -> Retriever:
    if name not in RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}; available: {sorted(RETRIEVERS)}")
    return RETRIEVERS[name](params or {})


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
    "DenseRetriever",
    "HybridRetriever",
    "RerankRetriever",
    "build_retriever",
    "build_reranker",
    "tokenize",
]
