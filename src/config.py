"""Config loading and validation.

Every pipeline choice lives here rather than in code, because the improvement loop
rewrites these values (ADR-002). Unknown keys are a hard error: a typo must never
silently fall back to a default and produce a quietly different experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

MetricMode = Literal["paper", "union"]
ReadMode = Literal["raw", "text"]
Split = Literal["train", "test", "all"]

SPLITS: tuple[str, ...] = ("train", "test", "all")
METRIC_MODES: tuple[str, ...] = ("paper", "union")
READ_MODES: tuple[str, ...] = ("raw", "text")


class ConfigError(ValueError):
    """Raised when a config is structurally invalid."""


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class DataSpec:
    benchmark: str = "cuad"
    mini_limit: int = 194
    read_mode: ReadMode = "raw"
    split: Split = "train"


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """Query-time text handling, kept separate so it can be ablated."""

    strip_doc_hint: bool = False


@dataclass(frozen=True, slots=True)
class MetricsSpec:
    mode: MetricMode = "paper"
    ks: tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    name: str
    chunking: StageSpec
    retrieval: StageSpec
    abstain: StageSpec
    data: DataSpec = field(default_factory=DataSpec)
    rerank: StageSpec = field(default_factory=lambda: StageSpec("none"))
    query: QuerySpec = field(default_factory=QuerySpec)
    metrics: MetricsSpec = field(default_factory=MetricsSpec)
    seed: int = 13

    @property
    def top_k(self) -> int:
        return self.retrieval.top_k or 10

    @property
    def max_k(self) -> int:
        return max(self.metrics.ks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "data": {
                "benchmark": self.data.benchmark,
                "mini_limit": self.data.mini_limit,
                "read_mode": self.data.read_mode,
                "split": self.data.split,
            },
            "chunking": {
                "name": self.chunking.name,
                "params": self.chunking.params,
            },
            "retrieval": {
                "name": self.retrieval.name,
                "params": self.retrieval.params,
                "top_k": self.top_k,
            },
            "rerank": {"name": self.rerank.name, "params": self.rerank.params},
            "abstain": {"name": self.abstain.name, "params": self.abstain.params},
            "query": {"strip_doc_hint": self.query.strip_doc_hint},
            "metrics": {"mode": self.metrics.mode, "ks": list(self.metrics.ks)},
        }

    def hash(self) -> str:
        """Stable short hash, used in cache keys and run IDs."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _expect_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {where}: {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def _stage(raw: Any, where: str, *, allow_top_k: bool = False) -> StageSpec:
    mapping = _expect_mapping(raw, where)
    allowed = {"name", "params"} | ({"top_k"} if allow_top_k else set())
    _reject_unknown(mapping, allowed, where)

    name = mapping.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}.name must be a non-empty string")

    params = _expect_mapping(mapping.get("params") or {}, f"{where}.params")
    top_k = mapping.get("top_k")
    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ConfigError(f"{where}.top_k must be a positive integer")
    return StageSpec(name=name, params=dict(params), top_k=top_k)


def parse_config(raw: dict[str, Any]) -> PipelineConfig:
    """Validate a raw mapping and build a :class:`PipelineConfig`."""
    allowed = {
        "name",
        "seed",
        "data",
        "chunking",
        "retrieval",
        "rerank",
        "abstain",
        "query",
        "metrics",
    }
    _reject_unknown(raw, allowed, "config root")

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("config.name must be a non-empty string")

    for required in ("chunking", "retrieval", "abstain"):
        if required not in raw:
            raise ConfigError(f"missing required section: {required}")

    data_raw = _expect_mapping(raw.get("data") or {}, "data")
    _reject_unknown(data_raw, {"benchmark", "mini_limit", "read_mode", "split"}, "data")
    split = data_raw.get("split", "train")
    if split not in SPLITS:
        raise ConfigError(f"data.split must be one of {SPLITS}, got {split!r}")
    read_mode = data_raw.get("read_mode", "raw")
    if read_mode not in READ_MODES:
        raise ConfigError(f"data.read_mode must be one of {READ_MODES}, got {read_mode!r}")
    mini_limit = data_raw.get("mini_limit", 194)
    if not isinstance(mini_limit, int) or mini_limit < 0:
        raise ConfigError("data.mini_limit must be a non-negative integer")

    query_raw = _expect_mapping(raw.get("query") or {}, "query")
    _reject_unknown(query_raw, {"strip_doc_hint"}, "query")
    strip_doc_hint = query_raw.get("strip_doc_hint", False)
    if not isinstance(strip_doc_hint, bool):
        raise ConfigError("query.strip_doc_hint must be a boolean")

    metrics_raw = _expect_mapping(raw.get("metrics") or {}, "metrics")
    _reject_unknown(metrics_raw, {"mode", "ks"}, "metrics")
    mode = metrics_raw.get("mode", "paper")
    if mode not in METRIC_MODES:
        raise ConfigError(f"metrics.mode must be one of {METRIC_MODES}, got {mode!r}")
    ks = metrics_raw.get("ks", [1, 2, 4, 8, 16])
    if not isinstance(ks, list) or not ks or not all(isinstance(k, int) and k > 0 for k in ks):
        raise ConfigError("metrics.ks must be a non-empty list of positive integers")

    seed = raw.get("seed", 13)
    if not isinstance(seed, int):
        raise ConfigError("seed must be an integer")

    return PipelineConfig(
        name=name,
        seed=seed,
        data=DataSpec(
            benchmark=data_raw.get("benchmark", "cuad"),
            mini_limit=mini_limit,
            read_mode=read_mode,  # type: ignore[arg-type]
            split=split,  # type: ignore[arg-type]
        ),
        chunking=_stage(raw["chunking"], "chunking"),
        retrieval=_stage(raw["retrieval"], "retrieval", allow_top_k=True),
        rerank=_stage(raw.get("rerank") or {"name": "none"}, "rerank"),
        abstain=_stage(raw["abstain"], "abstain"),
        query=QuerySpec(strip_doc_hint=strip_doc_hint),
        metrics=MetricsSpec(mode=mode, ks=tuple(ks)),  # type: ignore[arg-type]
    )


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return parse_config(raw)
