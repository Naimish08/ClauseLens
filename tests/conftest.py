"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def baseline_dict() -> dict[str, Any]:
    """A minimal valid pipeline config, matching configs/baseline.yaml's shape."""
    return {
        "name": "baseline-test",
        "seed": 13,
        "data": {
            "benchmark": "cuad",
            "mini_limit": 194,
            "read_mode": "raw",
            "split": "train",
        },
        "chunking": {"name": "fixed", "params": {"size": 512, "overlap": 0}},
        "retrieval": {"name": "bm25", "params": {"k1": 1.5, "b": 0.75}, "top_k": 10},
        "rerank": {"name": "none", "params": {}},
        "abstain": {"name": "none", "params": {}},
        "query": {"strip_doc_hint": False},
        "metrics": {"mode": "paper", "ks": [1, 2, 4, 8, 16]},
    }
