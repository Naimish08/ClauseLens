"""Tests for the improvement loop: diagnosis, proposals and acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from loop.accept import evaluate_gates
from loop.diagnose import diagnose
from loop.propose import (
    Proposal,
    behaviour_signature,
    propose,
)
from loop.run_loop import main as loop_main
from src.config import parse_config
from src.metrics import QueryScore


def score(
    recall: float,
    *,
    precision: float = 0.1,
    answerable: bool = True,
    abstained: bool = False,
    retrieved_len: int = 1000,
    failure: str | None = None,
) -> QueryScore:
    return QueryScore(
        query="q",
        category="c",
        split="train",
        answerable=answerable,
        abstained=abstained,
        k=10,
        n_retrieved=0 if abstained else 10,
        retrieved_len=retrieved_len,
        overlap_len=int(recall * 100),
        gt_len=100,
        recall=recall,
        precision=precision,
        f1=0.0,
        doc_hit=True,
        span_hit=recall > 0,
        chunk_hit=recall >= 1.0,
        mrr=0.5,
    )


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #
def test_diagnose_counts_and_weights_by_loss() -> None:
    scores = [
        score(0.0, failure="wrong_section"),
        score(0.5, failure="partial_span"),
        score(1.0, failure=None),
    ]
    diagnosis = diagnose(scores)
    assert diagnosis.counts["wrong_section"] == 1
    assert diagnosis.counts["partial_span"] == 1
    assert diagnosis.loss["wrong_section"] == pytest.approx(1.0)
    assert diagnosis.loss["partial_span"] == pytest.approx(0.5)
    assert diagnosis.dominant == "wrong_section"
    assert diagnosis.share("wrong_section") == pytest.approx(2 / 3)


def test_diagnose_with_no_failures() -> None:
    diagnosis = diagnose([score(1.0, precision=1.0)])
    assert diagnosis.dominant is None
    assert diagnosis.total_loss == 0.0


# --------------------------------------------------------------------------- #
# Proposals
# --------------------------------------------------------------------------- #
def _config(**overrides: Any):
    raw = {
        "name": "t",
        "data": {"split": "train"},
        "chunking": {"name": "fixed", "params": {"size": 512, "overlap": 0}},
        "retrieval": {"name": "bm25", "params": {}, "top_k": 10},
        "abstain": {"name": "none"},
        "query": {"strip_doc_hint": False},
        "metrics": {"mode": "paper", "ks": [1, 10]},
    }
    raw.update(overrides)
    return parse_config(raw)


def test_size_up_doubles_chunk_size() -> None:
    proposal = propose(_config(), "wrong_section", set(), set())
    assert isinstance(proposal, Proposal)
    assert proposal.rule == "size_up"
    assert proposal.config.chunking.params["size"] == 1024
    assert proposal.diff["chunking.params.size"] == {"from": 512, "to": 1024}


def test_size_up_stops_at_the_cap() -> None:
    config = _config(chunking={"name": "fixed", "params": {"size": 4096, "overlap": 0}})
    proposal = propose(config, "wrong_section", set(), set())
    # size_up returns None, so the next rule fires.
    assert proposal is not None
    assert proposal.rule != "size_up"


def test_size_down_reduces_size_with_a_floor() -> None:
    proposal = propose(_config(), "extra_context", set(), set())
    assert proposal is not None and proposal.rule == "size_down"
    assert proposal.config.chunking.params["size"] == 256


def test_size_down_floors_at_128() -> None:
    config = _config(chunking={"name": "fixed", "params": {"size": 128, "overlap": 0}})
    # No shrinks remain and extra_context has no other rule.
    assert propose(config, "extra_context", set(), set()) is None


def test_clause_proposal_maps_size_to_max_size() -> None:
    proposal = propose(_config(), "partial_span", {"size_up"}, set())
    assert proposal is not None and proposal.rule == "chunker_clause"
    assert proposal.config.chunking.params["max_size"] == 512


def test_sentence_proposal_keeps_size() -> None:
    proposal = propose(_config(), "wrong_section", {"size_up", "chunker_recursive"}, set())
    assert proposal is not None and proposal.rule == "chunker_sentence"
    assert proposal.config.chunking.params["size"] == 512


def test_tried_rules_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loop.propose.ml_available", lambda: False)
    tried = {
        "size_up",
        "chunker_recursive",
        "chunker_sentence",
        "chunker_clause",
        "overlap_on",
    }
    # Only the ML-backed rules remain, and they are unavailable in this test.
    proposal = propose(_config(), "wrong_section", tried, set())
    assert proposal is None


def test_hybrid_rule_fires_when_ml_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loop.propose.ml_available", lambda: True)
    tried = {
        "size_up",
        "chunker_recursive",
        "chunker_sentence",
        "chunker_clause",
        "overlap_on",
    }
    proposal = propose(_config(), "wrong_section", tried, set())
    assert proposal is not None and proposal.rule == "retriever_hybrid"
    assert proposal.config.retrieval.name == "hybrid"


def test_tried_signatures_are_skipped() -> None:
    first = propose(_config(), "wrong_section", set(), set())
    assert first is not None
    tried = {behaviour_signature(first.config), behaviour_signature(_config())}
    second = propose(_config(), "wrong_section", set(), tried)
    # The identical behaviour is skipped; a different rule may still fire.
    assert second is None or second.rule != first.rule


def test_no_dominant_failure_means_no_proposal() -> None:
    assert propose(_config(), None, set(), set()) is None


def test_strip_doc_hint_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loop.propose.ml_available", lambda: False)
    proposal = propose(_config(), "wrong_document", {"retriever_hybrid"}, set())
    assert proposal is None or proposal.rule == "strip_doc_hint"
    if proposal is not None:
        assert proposal.config.query.strip_doc_hint is True


def test_behaviour_signature_ignores_the_name() -> None:
    a = _config()
    b = parse_config({**a.as_dict(), "name": "other"})
    assert behaviour_signature(a) == behaviour_signature(b)


# --------------------------------------------------------------------------- #
# Acceptance
# --------------------------------------------------------------------------- #
def _aligned(current_recall: list[float], candidate_recall: list[float], **kwargs):
    current = [
        score(value, failure="wrong_section" if value < 1 else None)
        for value in current_recall
    ]
    candidate = [score(value) for value in candidate_recall]
    folds = [index % 5 for index in range(len(current))]
    return evaluate_gates(current, candidate, folds, **kwargs)


def test_clear_improvement_is_accepted() -> None:
    result = _aligned([0.2] * 20, [0.4] * 20)
    assert result.accepted
    assert result.delta == pytest.approx(0.2)
    assert result.consistency


def test_small_improvement_is_rejected() -> None:
    result = _aligned([0.2] * 20, [0.205] * 20)
    assert not result.accepted
    assert not result.effect_size


def test_improvement_in_one_fold_is_rejected() -> None:
    current = [0.2] * 20
    candidate = [0.2] * 20
    for index in [0, 1, 2]:
        candidate[index] = 0.9
    result = _aligned(current, candidate)
    assert not result.accepted
    assert not result.consistency


def test_returning_more_text_is_a_guardrail_failure() -> None:
    current = [score(0.2, retrieved_len=1000) for _ in range(20)]
    candidate = [score(0.4, retrieved_len=5000) for _ in range(20)]
    folds = [index % 5 for index in range(20)]
    result = evaluate_gates(
        current,
        candidate,
        folds,
        baseline_mean_retrieved_len=1000.0,
    )
    assert not result.accepted
    assert not result.guardrails
    assert "retrieved" in result.notes


def test_precision_collapse_is_a_guardrail_failure() -> None:
    current = [score(0.2, precision=0.10) for _ in range(20)]
    candidate = [score(0.4, precision=0.01) for _ in range(20)]
    folds = [index % 5 for index in range(20)]
    result = evaluate_gates(
        current,
        candidate,
        folds,
        baseline_mean_retrieved_len=1000.0,
    )
    assert not result.accepted
    assert not result.guardrails


def test_misaligned_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="aligned"):
        evaluate_gates([score(0.1)], [score(0.2), score(0.3)], [0, 1])


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_loop_writes_a_log_with_rejections(tmp_path: Path) -> None:
    log_path = tmp_path / "experiments.jsonl"
    config_dir = tmp_path / "generated"
    code = loop_main(
        [
            "--config",
            "configs/baseline.yaml",
            "--iterations",
            "2",
            "--limit",
            "12",
            "--log-path",
            str(log_path),
            "--config-dir",
            str(config_dir),
        ]
    )
    assert code == 0

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0]["iteration"] == 0
    assert records[0]["proposal"]["proposer"] == "baseline"
    assert len(records) >= 2
    for record in records[1:]:
        assert {"iteration", "accepted", "proposal", "evaluation", "decision"} <= set(record)
        assert record["evaluation"]["split"] == "train"
    # With 12 queries the intervals are wide, so the point of this test is the
    # log shape and that rejections are recorded, not that anything is accepted.
    assert any(record["accepted"] is False for record in records[1:])
    for record in records[1:]:
        assert record["decision"]["gate_significance"] in {True, False}
