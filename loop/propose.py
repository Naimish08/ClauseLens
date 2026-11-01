"""Deterministic proposal rules for the improvement loop.

No LLM: the dominant failure maps to an ordered list of candidate changes, and each
candidate is a concrete config, not prose. Proposals that need the ML stack are
skipped when it is not installed, so the loop still runs on a bare environment.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from dataclasses import dataclass
from typing import Any

from src.config import PipelineConfig, StageSpec

RULES: dict[str, tuple[str, ...]] = {
    # The mapping recorded in docs/07-improvement-loop.md.
    "wrong_section": (
        "size_up",
        "chunker_recursive",
        "chunker_sentence",
        "chunker_clause",
        "overlap_on",
        "retriever_hybrid",
        "rerank",
    ),
    "partial_span": ("size_up", "chunker_clause", "chunker_recursive", "overlap_on"),
    "extra_context": ("size_down", "overlap_off"),
    "wrong_document": ("retriever_hybrid", "strip_doc_hint", "size_up"),
    "false_abstention": ("abstain_relax",),
    "missed_abstention": ("abstain_tighten",),
}

RULE_DESCRIPTIONS: dict[str, str] = {
    "size_up": "increase chunk size (more chance a GT span sits inside one chunk)",
    "size_down": "decrease chunk size (reduce returned text and improve precision)",
    "chunker_recursive": "switch to recursive character splitting at the current size",
    "chunker_sentence": "switch to sentence-window chunking at the current size",
    "chunker_clause": "switch to clause-aware chunking at the current size",
    "retriever_hybrid": "fuse BM25 with a dense retriever using reciprocal rank fusion",
    "rerank": "wrap the current retriever in a cross-encoder reranker",
    "overlap_on": "add 15% chunk overlap (rescue answers that straddle a boundary)",
    "overlap_off": "remove chunk overlap (stop returning duplicated text)",
    "strip_doc_hint": "strip the 'Consider the ...' prefix from the query",
    "abstain_relax": "lower the abstention threshold",
    "abstain_tighten": "raise the abstention threshold",
}


@dataclass(frozen=True, slots=True)
class Proposal:
    rule: str
    reason: str
    config: PipelineConfig
    diff: dict[str, dict[str, Any]]


def ml_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("sentence_transformers") is not None
    )


def _diff(before: PipelineConfig, after: PipelineConfig) -> dict[str, dict[str, Any]]:
    """Flat diff of the behaviour-bearing config keys, old -> new."""

    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, dict):
            flat: dict[str, Any] = {}
            for key, item in value.items():
                flat.update(flatten(item, f"{prefix}.{key}" if prefix else key))
            return flat
        return {prefix: value}

    old = flatten(before.as_dict())
    new = flatten(after.as_dict())
    return {
        key: {"from": old.get(key), "to": new.get(key)}
        for key in sorted(set(old) | set(new))
        if old.get(key) != new.get(key)
    }


def _size_key(params: dict[str, Any]) -> str:
    return "max_size" if "max_size" in params else "size"


def _with_chunking(config: PipelineConfig, name: str, params: dict[str, Any]) -> PipelineConfig:
    return dataclasses.replace(
        config, chunking=dataclasses.replace(config.chunking, name=name, params=params)
    )


def _size_up(config: PipelineConfig) -> PipelineConfig | None:
    params = dict(config.chunking.params)
    key = _size_key(params)
    current = int(params.get(key) or 512)
    new_size = min(current * 2, 4096)
    if new_size == current:
        return None
    params[key] = new_size
    # Only rescale an overlap that already exists: turning overlap on at the same
    # time would confound two changes in one iteration.
    if int(params.get("overlap", 0)) > 0:
        params["overlap"] = int(new_size * 0.15)
    return _with_chunking(config, config.chunking.name, params)


def _size_down(config: PipelineConfig) -> PipelineConfig | None:
    params = dict(config.chunking.params)
    key = _size_key(params)
    current = int(params.get(key) or 512)
    new_size = max(current // 2, 128)
    if new_size == current:
        return None
    params[key] = new_size
    if int(params.get("overlap", 0)) > 0:
        params["overlap"] = int(new_size * 0.15)
    return _with_chunking(config, config.chunking.name, params)


def _current_size(config: PipelineConfig) -> int:
    params = config.chunking.params
    return int(params.get(_size_key(params)) or 512)


def _chunker_recursive(config: PipelineConfig) -> PipelineConfig | None:
    if config.chunking.name == "recursive":
        return None
    return _with_chunking(
        config, "recursive", {"size": _current_size(config), "overlap": 0}
    )


def _overlap_on(config: PipelineConfig) -> PipelineConfig | None:
    params = dict(config.chunking.params)
    size = _current_size(config)
    if int(params.get("overlap", 0)) > 0:
        return None
    params["overlap"] = max(1, int(size * 0.15))
    return _with_chunking(config, config.chunking.name, params)


def _overlap_off(config: PipelineConfig) -> PipelineConfig | None:
    params = dict(config.chunking.params)
    if int(params.get("overlap", 0)) == 0:
        return None
    params["overlap"] = 0
    return _with_chunking(config, config.chunking.name, params)


def _chunker_sentence(config: PipelineConfig) -> PipelineConfig | None:
    if config.chunking.name == "sentence":
        return None
    return _with_chunking(
        config, "sentence", {"size": _current_size(config), "overlap": 0}
    )


def _chunker_clause(config: PipelineConfig) -> PipelineConfig | None:
    if config.chunking.name == "clause":
        return None
    size = _current_size(config)
    return _with_chunking(
        config, "clause", {"max_size": size, "min_size": min(200, size), "overlap": 0}
    )


def _retriever_hybrid(config: PipelineConfig) -> PipelineConfig | None:
    if not ml_available():
        return None
    if config.retrieval.name in {"hybrid", "rerank"}:
        return None
    dense = {
        "name": "dense",
        "params": {"model": "bge-base-en-v1.5"},
    }
    return dataclasses.replace(
        config,
        retrieval=dataclasses.replace(
            config.retrieval,
            name="hybrid",
            params={
                "members": [
                    {"name": config.retrieval.name, "params": dict(config.retrieval.params)},
                    dense,
                ],
                "weights": [1.0, 1.0],
                "k_rrf": 60,
                "depth": 100,
            },
        ),
    )


def _rerank(config: PipelineConfig) -> PipelineConfig | None:
    if not ml_available():
        return None
    if config.retrieval.name == "rerank":
        return None
    base = {"name": config.retrieval.name, "params": dict(config.retrieval.params)}
    return dataclasses.replace(
        config,
        retrieval=dataclasses.replace(
            config.retrieval,
            name="rerank",
            params={
                "base": base,
                "model": "BAAI/bge-reranker-base",
                "candidates": 50,
            },
        ),
    )


def _strip_doc_hint(config: PipelineConfig) -> PipelineConfig | None:
    if config.query.strip_doc_hint:
        return None
    return dataclasses.replace(
        config,
        query=dataclasses.replace(config.query, strip_doc_hint=True),
    )


def _abstain_relax(config: PipelineConfig) -> PipelineConfig | None:
    threshold = float(config.abstain.params.get("threshold", 0.5))
    lowered = max(0.0, threshold / 2)
    if lowered == threshold:
        return None
    return dataclasses.replace(
        config,
        abstain=StageSpec(name="threshold", params={"threshold": lowered}, top_k=None),
    )


def _abstain_tighten(config: PipelineConfig) -> PipelineConfig | None:
    threshold = float(config.abstain.params.get("threshold", 0.5))
    raised = threshold * 2 if threshold > 0 else 1.0
    return dataclasses.replace(
        config,
        abstain=StageSpec(name="threshold", params={"threshold": raised}, top_k=None),
    )


BUILDERS = {
    "size_up": _size_up,
    "size_down": _size_down,
    "chunker_recursive": _chunker_recursive,
    "chunker_sentence": _chunker_sentence,
    "overlap_on": _overlap_on,
    "overlap_off": _overlap_off,
    "chunker_clause": _chunker_clause,
    "retriever_hybrid": _retriever_hybrid,
    "rerank": _rerank,
    "strip_doc_hint": _strip_doc_hint,
    "abstain_relax": _abstain_relax,
    "abstain_tighten": _abstain_tighten,
}


def behaviour_signature(config: PipelineConfig) -> str:
    """Hash of the behaviour-bearing config, ignoring the display name."""
    payload = config.as_dict()
    payload.pop("name", None)
    import json

    return json.dumps(payload, sort_keys=True)


def propose(
    config: PipelineConfig,
    dominant: str | None,
    tried_rules: set[str],
    tried_signatures: set[str],
) -> Proposal | None:
    """Return the next untried candidate for the dominant failure, if any."""
    if dominant is None:
        return None

    for rule in RULES.get(dominant, ()):
        if rule in tried_rules:
            continue
        builder = BUILDERS.get(rule)
        if builder is None:
            continue
        candidate = builder(config)
        if candidate is None:
            tried_rules.add(rule)
            continue
        if behaviour_signature(candidate) in tried_signatures:
            tried_rules.add(rule)
            continue
        return Proposal(
            rule=rule,
            reason=f"{dominant} is dominant: {RULE_DESCRIPTIONS[rule]}",
            config=candidate,
            diff=_diff(config, candidate),
        )
    return None
