"""Tests for config validation and the pipeline wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.config import ConfigError, load_config, parse_config
from src.data import Document
from src.pipeline import Pipeline


def test_baseline_config_loads() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "baseline.yaml")
    assert config.name == "baseline-fixed512-bm25"
    assert config.chunking.name == "fixed"
    assert config.chunking.params["size"] == 512
    assert config.retrieval.name == "bm25"
    assert config.top_k == 10
    assert config.data.split == "train"


def test_unknown_root_key_is_rejected(baseline_dict: dict[str, Any]) -> None:
    baseline_dict["chunk_size"] = 512
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(baseline_dict)


def test_unknown_stage_key_is_rejected(baseline_dict: dict[str, Any]) -> None:
    baseline_dict["chunking"]["strategy"] = "fixed"
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(baseline_dict)


def test_missing_required_section_is_rejected(baseline_dict: dict[str, Any]) -> None:
    del baseline_dict["retrieval"]
    with pytest.raises(ConfigError, match="missing required section"):
        parse_config(baseline_dict)


def test_invalid_split_is_rejected(baseline_dict: dict[str, Any]) -> None:
    baseline_dict["data"]["split"] = "validation"
    with pytest.raises(ConfigError, match="data.split"):
        parse_config(baseline_dict)


def test_invalid_metric_mode_is_rejected(baseline_dict: dict[str, Any]) -> None:
    baseline_dict["metrics"] = {"mode": "cosine"}
    with pytest.raises(ConfigError, match="metrics.mode"):
        parse_config(baseline_dict)


def test_config_hash_is_stable_and_sensitive(baseline_dict: dict[str, Any]) -> None:
    first = parse_config(baseline_dict)
    second = parse_config(baseline_dict)
    assert first.hash() == second.hash()

    baseline_dict["chunking"]["params"]["size"] = 256
    assert parse_config(baseline_dict).hash() != first.hash()


def test_pipeline_indexes_and_searches(baseline_dict: dict[str, Any]) -> None:
    pipeline = Pipeline.build(parse_config(baseline_dict))
    documents = [
        Document("cuad/a.txt", "The governing law of this agreement is Delaware."),
        Document("cuad/b.txt", "The parties shall maintain confidentiality of information."),
    ]
    stats = pipeline.index(documents)
    assert stats.documents == 2
    assert stats.chunks >= 2

    result = pipeline.search("governing law")
    assert not result.abstained
    assert result.spans
    assert result.spans[0].file_path == "cuad/a.txt"
    assert result.timings_ms["retrieval"] >= 0.0


def test_pipeline_search_before_index_is_an_error(baseline_dict: dict[str, Any]) -> None:
    pipeline = Pipeline.build(parse_config(baseline_dict))
    with pytest.raises(RuntimeError, match="before index"):
        pipeline.search("anything")


def test_pipeline_abstains_when_nothing_matches(baseline_dict: dict[str, Any]) -> None:
    pipeline = Pipeline.build(parse_config(baseline_dict))
    pipeline.index([Document("cuad/a.txt", "the governing law is delaware")])
    result = pipeline.search("zzzzz unmatched token")
    assert result.abstained
    assert result.spans == ()


def test_strip_doc_hint_is_off_by_default_and_strips_when_enabled(
    baseline_dict: dict[str, Any],
) -> None:
    plain = Pipeline.build(parse_config(baseline_dict))
    query = "Consider the X Agreement; What is the governing law?"
    assert plain.prepare_query(query) == query

    baseline_dict["query"] = {"strip_doc_hint": True}
    stripped = Pipeline.build(parse_config(baseline_dict))
    assert stripped.prepare_query(query) == "What is the governing law?"


def test_pipeline_returns_only_indexed_chunk_ids(baseline_dict: dict[str, Any]) -> None:
    pipeline = Pipeline.build(parse_config(baseline_dict))
    pipeline.index(
        [
            Document("cuad/a.txt", "governing law delaware " * 40),
            Document("cuad/b.txt", "confidentiality information " * 40),
        ]
    )
    result = pipeline.search("governing law")
    assert result.spans
    for span in result.spans:
        assert span.chunk_id in pipeline.indexed_chunk_ids()


def test_config_round_trips_through_as_dict(baseline_dict: dict[str, Any]) -> None:
    config = parse_config(baseline_dict)
    assert parse_config(config.as_dict()).hash() == config.hash()
