"""Tests for retrieval registry composition and the retrieval grid config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.run_retrieval_grid import variant_config
from src.retrieval import build_retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import RerankRetriever

GRID = Path(__file__).parents[1] / "configs" / "grid_retrieval.yaml"


def grid() -> dict:
    return yaml.safe_load(GRID.read_text(encoding="utf-8"))


def test_every_variant_builds_a_valid_config() -> None:
    spec = grid()
    names = [variant["name"] for variant in spec["variants"]]
    assert "bm25" in names and "hybrid-rerank" in names
    for variant in spec["variants"]:
        config = variant_config(spec, variant)
        assert config.name == variant["name"]
        assert config.chunking.name == "sentence"  # held fixed
        assert config.data.split == "train"


def test_dense_registry_builds_lazily() -> None:
    retriever = build_retriever("dense", {"model": "gte-base"})
    assert isinstance(retriever, DenseRetriever)
    assert retriever.model == "gte-base"
    # Constructing must not download or import torch; indexing would.
    assert len(retriever) == 0


def test_hybrid_registry_builds_members() -> None:
    retriever = build_retriever(
        "hybrid",
        {
            "members": [
                {"name": "bm25", "params": {"k1": 1.5, "b": 0.75}},
                {"name": "dense", "params": {"model": "gte-base"}},
            ],
            "weights": [2.0, 1.0],
            "k_rrf": 60,
            "depth": 100,
        },
    )
    assert isinstance(retriever, HybridRetriever)
    assert len(retriever.members) == 2
    assert retriever.weights == [2.0, 1.0]
    assert retriever.k_rrf == 60


def test_hybrid_registry_requires_members() -> None:
    with pytest.raises(ValueError, match="members"):
        build_retriever("hybrid", {})


def test_rerank_registry_wraps_a_base() -> None:
    retriever = build_retriever(
        "rerank",
        {
            "base": {"name": "bm25", "params": {"k1": 1.5, "b": 0.75}},
            "candidates": 20,
            "model": "BAAI/bge-reranker-base",
        },
    )
    assert isinstance(retriever, RerankRetriever)
    assert retriever.candidates == 20
    assert retriever.base.name == "bm25"


def test_rerank_registry_requires_a_base() -> None:
    with pytest.raises(ValueError, match="base"):
        build_retriever("rerank", {})


def test_unknown_nested_member_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever(
            "hybrid", {"members": [{"name": "quantum", "params": {}}]}
        )


def test_unknown_embedder_is_rejected() -> None:
    from src.retrieval.dense import build_encoder

    with pytest.raises(ValueError, match="unknown embedder"):
        build_encoder("text-embedding-4")
