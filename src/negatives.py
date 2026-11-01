"""The frozen unanswerable (negative) query set.

Every entry is a query the indexed corpus does not answer, with machine-checkable
evidence for *why* it is unanswerable. The set is committed once and never edited
(AGENTS.md rule 2); the commit history is the evidence of the freeze.

Four types (``docs/06-abstention.md``):

1. clause type absent from every indexed contract
2. a contract that is not in the corpus
3. clause type that exists in the corpus, but not in the queried contract
4. near-miss: a related clause exists, the asked-for one does not

``absent_patterns`` are regular expressions that must **not** match inside
``scope`` — ``scope="corpus"`` for types 1 and 2, a specific document for types 3
and 4. ``present_patterns`` (type 4 only) must match inside the scope, proving the
near-miss neighbour really exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data import Document, assign_split

NEGATIVE_TYPES: tuple[int, ...] = (1, 2, 3, 4)
TYPE_DESCRIPTIONS = {
    1: "clause type absent from every indexed contract",
    2: "contract not present in the corpus",
    3: "clause exists in the corpus but not in this contract",
    4: "near-miss vocabulary: related clause exists, asked-for clause does not",
}
CORPUS_SCOPE = "corpus"


@dataclass(frozen=True, slots=True)
class NegativeQuery:
    id: str
    query: str
    type: int
    rationale: str
    scope: str
    absent_patterns: tuple[str, ...]
    present_patterns: tuple[str, ...] = ()
    related_clause: str = ""
    verified_by: str = ""

    @property
    def split(self) -> str:
        return assign_split(self.query)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "query": self.query,
            "type": self.type,
            "split": self.split,
            "scope": self.scope,
            "absent_patterns": list(self.absent_patterns),
            "rationale": self.rationale,
            "verified_by": self.verified_by,
        }
        if self.present_patterns:
            payload["present_patterns"] = list(self.present_patterns)
        if self.related_clause:
            payload["related_clause"] = self.related_clause
        return payload


@dataclass
class Violation:
    negative_id: str
    problem: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.negative_id}: {self.problem}" + (f" ({self.detail})" if self.detail else "")


@dataclass
class ValidationReport:
    entries: int
    per_type: dict[int, int] = field(default_factory=dict)
    per_split: dict[str, int] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        return (
            f"entries={self.entries} per_type={self.per_type} "
            f"per_split={self.per_split} violations={len(self.violations)}"
        )


def load_negatives(path: Path) -> list[NegativeQuery]:
    entries: list[NegativeQuery] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            entries.append(
                NegativeQuery(
                    id=raw["id"],
                    query=raw["query"],
                    type=int(raw["type"]),
                    rationale=raw.get("rationale", ""),
                    scope=raw.get("scope", CORPUS_SCOPE),
                    absent_patterns=tuple(raw.get("absent_patterns", ())),
                    present_patterns=tuple(raw.get("present_patterns", ())),
                    related_clause=raw.get("related_clause", ""),
                    verified_by=raw.get("verified_by", ""),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{line_number} missing field {exc}") from exc
    return entries


def save_negatives(entries: list[NegativeQuery], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry.as_dict(), ensure_ascii=False) for entry in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def validate_negatives(
    entries: list[NegativeQuery],
    documents: dict[str, Document],
    *,
    min_entries: int = 30,
) -> ValidationReport:
    """Check the set's internal consistency and its claims about the corpus."""
    report = ValidationReport(entries=len(entries))
    corpus_text = "\n".join(document.content for document in documents.values())

    seen_ids: set[str] = set()
    for entry in entries:
        report.per_type[entry.type] = report.per_type.get(entry.type, 0) + 1
        report.per_split[entry.split] = report.per_split.get(entry.split, 0) + 1

        if entry.id in seen_ids:
            report.violations.append(Violation(entry.id, "duplicate id"))
        seen_ids.add(entry.id)

        if entry.type not in NEGATIVE_TYPES:
            report.violations.append(Violation(entry.id, f"invalid type {entry.type}"))
        if len(entry.query.strip()) < 20:
            report.violations.append(Violation(entry.id, "query too short to be meaningful"))
        if not entry.absent_patterns:
            report.violations.append(Violation(entry.id, "no absent_patterns to verify"))

        if entry.scope == CORPUS_SCOPE:
            scope_text = corpus_text
        elif entry.scope in documents:
            scope_text = documents[entry.scope].content
        else:
            report.violations.append(Violation(entry.id, f"unknown scope {entry.scope!r}"))
            continue

        found = _matches(entry.absent_patterns, scope_text)
        if found:
            report.violations.append(
                Violation(entry.id, "absent pattern present in scope", ", ".join(found))
            )

        if entry.type == 4:
            if not entry.present_patterns:
                report.violations.append(
                    Violation(entry.id, "type 4 entry needs present_patterns")
                )
            elif not _matches(entry.present_patterns, scope_text):
                report.violations.append(
                    Violation(
                        entry.id,
                        "near-miss neighbour missing from scope",
                        ", ".join(entry.present_patterns),
                    )
                )

        if entry.type == 2 and entry.scope != CORPUS_SCOPE:
            report.violations.append(
                Violation(entry.id, "type 2 entries must be scoped to the corpus")
            )

    if len(entries) < min_entries:
        report.violations.append(
            Violation("<set>", f"only {len(entries)} entries, minimum is {min_entries}")
        )
    return report
