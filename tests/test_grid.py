"""Tests for the grid runner's config mapping and the containment diagnostic."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.run_grid import chunk_params, variant_config
from src.chunking import build_chunker
from src.data import Document, Query, Snippet
from src.metrics import span_containment
from src.types import make_chunk


def test_chunk_params_map_size_and_overlap_per_chunker() -> None:
    assert chunk_params("fixed", 512, 0) == {
        "size": 512,
        "overlap": 0,
        "snap": True,
        "snap_tolerance": 80,
    }
    assert chunk_params("recursive", 256, 38) == {"size": 256, "overlap": 38}
    assert chunk_params("sentence", 1024, 153) == {"size": 1024, "overlap": 153}
    clause = chunk_params("clause", 512, 76)
    assert clause["max_size"] == 512 and clause["overlap"] == 76


def test_chunk_params_reject_unknown_chunker() -> None:
    with pytest.raises(ValueError, match="unknown chunker"):
        chunk_params("semantic", 512, 0)


def _grid(chunker: str, size: int = 512, overlap: float = 0.15) -> dict[str, Any]:
    return {
        "seed": 13,
        "split": "train",
        "mode": "paper",
        "sizes": [size],
        "overlaps": [overlap],
        "chunkers": [chunker],
        "ks": [1, 10],
        "data": {"benchmark": "cuad", "mini_limit": 194, "read_mode": "raw"},
        "retrieval": {"name": "bm25", "params": {"k1": 1.5, "b": 0.75}, "top_k": 10},
    }


@pytest.mark.parametrize("chunker", ["fixed", "recursive", "sentence", "clause"])
def test_variant_config_is_valid_for_every_chunker(chunker: str) -> None:
    config = variant_config(_grid(chunker), chunker, 512, 0.15)
    assert config.chunking.name == chunker
    assert config.chunking.params["overlap"] == 76
    assert config.top_k == 10
    assert config.metrics.ks == (1, 10)
    # The resulting chunker must accept the generated parameters.
    assert build_chunker(config.chunking.name, config.chunking.params).name == chunker


def test_variant_name_encodes_the_axes() -> None:
    config = variant_config(_grid("fixed"), "fixed", 256, 0.0)
    assert config.name == "fixed-256-ov00"


# --------------------------------------------------------------------------- #
# Span containment
# --------------------------------------------------------------------------- #
def _docs() -> dict[str, Document]:
    return {"cuad/a.txt": Document("cuad/a.txt", "0123456789" * 10)}


def _chunks():
    text = _docs()["cuad/a.txt"].content
    return [
        make_chunk("cuad/a.txt", 0, 20, text[0:20]),
        make_chunk("cuad/a.txt", 20, 40, text[20:40]),
        make_chunk("cuad/a.txt", 40, 100, text[40:100]),
    ]


def test_containment_counts_fully_inside_spans() -> None:
    query = Query("q", (Snippet("cuad/a.txt", 50, 60, ""),))
    report = span_containment([query], _chunks())
    assert report.mean == 1.0
    assert report.full == 1 and report.partial == 1 and report.total == 1


def test_containment_detects_a_straddling_span() -> None:
    query = Query("q", (Snippet("cuad/a.txt", 15, 25, ""),))
    report = span_containment([query], _chunks())
    assert report.mean == 0.0
    assert report.full == 0
    assert report.partial == 1  # it is covered, just not by one chunk


def test_containment_detects_a_missing_span() -> None:
    query = Query("q", (Snippet("cuad/a.txt", 500, 510, ""),))
    report = span_containment([query], _chunks())
    assert report.mean == 0.0
    assert report.partial == 0


def test_containment_averages_over_queries() -> None:
    queries = [
        Query("q1", (Snippet("cuad/a.txt", 50, 60, ""),)),
        Query("q2", (Snippet("cuad/a.txt", 15, 25, ""),)),
    ]
    report = span_containment(queries, _chunks())
    assert report.mean == 0.5
    assert report.total == 2


def test_containment_of_no_queries_is_zero_not_an_error() -> None:
    report = span_containment([], _chunks())
    assert report.mean == 0.0 and report.total == 0
