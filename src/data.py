"""Data loading, mini-benchmark selection, splits and offset verification.

The benchmark is LegalBench-RAG (Pipitone & Houir Alami, 2024).  This module is
the *only* place that reads raw benchmark JSON or corpus text, so every stage
downstream sees identical bytes and identical spans.

Design notes
------------
* Spans are character offsets into the corpus file's decoded text.
* ``assign_split`` is a fixed, hash-derived split.  It must never be changed,
  because the reported test numbers depend on it.
* ``select_mini`` reproduces ``legalbenchrag/benchmark.py`` exactly, including
  its unusual per-test reseeding.  See ``docs/02-data.md`` for why.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
CORPUS_DIR = DATA_DIR / "corpus"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

#: The mini benchmark keeps at most this many queries per source benchmark.
MINI_LIMIT = 194

#: ``sha256(query) % SPLIT_MODULUS < TEST_BUCKETS`` -> test.
SPLIT_MODULUS = 10
TEST_BUCKETS = 3

Split = Literal["train", "test"]
ReadMode = Literal["text", "raw"]


# --------------------------------------------------------------------------- #
# Core records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Snippet:
    """A ground-truth answer span inside one corpus document."""

    file_path: str
    start: int
    end: int
    answer: str

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Query:
    """One benchmark question with its ground-truth snippets."""

    text: str
    snippets: tuple[Snippet, ...]
    tags: tuple[str, ...] = ()

    @property
    def split(self) -> Split:
        return assign_split(self.text)

    @property
    def category(self) -> str:
        """Clause type, derived from the question half of the query template."""
        return derive_category(self.text)

    @property
    def doc_hint(self) -> str:
        """The contract reference in the query prefix (empty if absent)."""
        return derive_doc_hint(self.text)

    @property
    def file_paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.file_path for s in self.snippets))

    @property
    def gt_length(self) -> int:
        return sum(s.length for s in self.snippets)

    def __len__(self) -> int:
        return len(self.snippets)


@dataclass(frozen=True, slots=True)
class Document:
    """A corpus document: its path relative to the corpus root, and its text."""

    file_path: str
    content: str

    def __len__(self) -> int:
        return len(self.content)


# --------------------------------------------------------------------------- #
# Query helpers
# --------------------------------------------------------------------------- #
def assign_split(query_text: str) -> Split:
    """Fixed hash-derived split.  Do not change: reported numbers depend on it."""
    digest = hashlib.sha256(query_text.strip().encode("utf-8")).hexdigest()
    return "test" if int(digest, 16) % SPLIT_MODULUS < TEST_BUCKETS else "train"


def _split_once(query_text: str) -> tuple[str, str]:
    prefix, sep, rest = query_text.partition(";")
    if not sep:
        return "", query_text.strip()
    return prefix.strip(), rest.strip()


def derive_category(query_text: str) -> str:
    """Return the clause-type question, e.g. ``What is the governing law ...?``.

    All 194 mini queries use ``"Consider the <contract>; <question>"``, so the
    text after the first semicolon is a stable, finite vocabulary of clause
    types (32 of them) suitable for a per-category breakdown.
    """
    return _split_once(query_text)[1]


def derive_doc_hint(query_text: str) -> str:
    """Return the ``"Consider the <contract>"`` prefix, or ``""`` if absent."""
    return _split_once(query_text)[0]


def split_counts(queries: Iterable[Query]) -> dict[str, int]:
    counts: dict[str, int] = {"train": 0, "test": 0}
    for query in queries:
        counts[query.split] += 1
    return counts


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_benchmark(name: str = "cuad", benchmarks_dir: Path | None = None) -> list[Query]:
    """Load a LegalBench-RAG benchmark JSON file into :class:`Query` records."""
    path = Path(benchmarks_dir or BENCHMARKS_DIR) / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: list[Query] = []
    for test in raw["tests"]:
        snippets = tuple(
            Snippet(
                file_path=s["file_path"],
                start=int(s["span"][0]),
                end=int(s["span"][1]),
                answer=s.get("answer", ""),
            )
            for s in test["snippets"]
        )
        queries.append(
            Query(text=test["query"], snippets=snippets, tags=tuple(test.get("tags") or ()))
        )
    return queries


def select_mini(queries: Sequence[Query], limit: int = MINI_LIMIT) -> list[Query]:
    """Replicate the reference ``mini`` selection from ``legalbenchrag/benchmark.py``.

    The reference sorts by a random key seeded from the *first snippet's file
    path*, which clusters the kept queries onto as few documents as possible
    (29 for CUAD) so ingestion is cheap.  Reproducing the reseeding exactly is
    required: the resulting span set is what every metric is measured against.
    """
    ordered = list(queries)
    if len(ordered) <= limit:
        return ordered

    def key(query: Query) -> float:
        random.seed(query.snippets[0].file_path)
        return random.random()

    return sorted(ordered, key=key)[:limit]


def read_document_text(path: str | Path, mode: ReadMode = "raw") -> str:
    """Read a corpus file.

    ``raw`` decodes bytes directly and preserves ``\\r\\n``; ``text`` uses
    Python's default universal-newline translation.  We support both because the
    benchmark's published offsets were produced by the reference implementation
    (which uses plain ``open()``), and the two disagree on CRLF files.
    """
    path = Path(path)
    if mode == "raw":
        return path.read_bytes().decode("utf-8")
    if mode == "text":
        return path.read_text(encoding="utf-8")
    raise ValueError(f"unknown read mode: {mode!r}")


def load_corpus(
    file_paths: Iterable[str] | None = None,
    corpus_dir: Path | None = None,
    mode: ReadMode = "raw",
) -> dict[str, Document]:
    """Load documents keyed by their corpus-relative path.

    ``file_paths`` defaults to every ``.txt`` file under the corpus root; the
    mini experiment passes only the 29 referenced paths.
    """
    root = Path(corpus_dir or CORPUS_DIR)
    if file_paths is None:
        paths = sorted(str(p.relative_to(root)) for p in root.rglob("*.txt"))
    else:
        paths = list(file_paths)
    return {fp: Document(file_path=fp, content=read_document_text(root / fp, mode)) for fp in paths}


def load_mini_cuad(
    mode: ReadMode = "raw", limit: int = MINI_LIMIT
) -> tuple[list[Query], dict[str, Document]]:
    """Load the CUAD-mini queries and exactly the documents they reference."""
    queries = select_mini(load_benchmark("cuad"), limit=limit)
    referenced = sorted({fp for q in queries for fp in q.file_paths})
    return queries, load_corpus(referenced, mode=mode)


def by_split(queries: Iterable[Query], split: Split) -> list[Query]:
    return [q for q in queries if q.split == split]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SpanProblem:
    query: str
    file_path: str
    span: tuple[int, int]
    kind: str  # "mismatch" | "out_of_bounds" | "missing_document"
    expected: str = ""
    actual: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "file_path": self.file_path,
            "span": list(self.span),
            "kind": self.kind,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class VerificationReport:
    mode: str
    queries: int
    snippets: int
    documents: int
    problems: list[SpanProblem] = field(default_factory=list)
    crlf_documents: tuple[str, ...] = ()
    overlapping_span_queries: tuple[str, ...] = ()

    @property
    def mismatches(self) -> list[SpanProblem]:
        return [p for p in self.problems if p.kind == "mismatch"]

    @property
    def out_of_bounds(self) -> list[SpanProblem]:
        return [p for p in self.problems if p.kind == "out_of_bounds"]

    @property
    def ok(self) -> bool:
        return not self.problems and not self.overlapping_span_queries

    def summary(self) -> str:
        return (
            f"mode={self.mode} queries={self.queries} snippets={self.snippets} "
            f"docs={self.documents} mismatches={len(self.mismatches)} "
            f"out_of_bounds={len(self.out_of_bounds)} "
            f"crlf_docs={len(self.crlf_documents)} "
            f"overlapping_gt_queries={len(self.overlapping_span_queries)}"
        )


def verify_offsets(
    queries: Sequence[Query],
    documents: dict[str, Document],
    *,
    mode: str = "raw",
    corpus_dir: Path | None = None,
) -> VerificationReport:
    """Check that ``document[start:end] == snippet.answer`` for every snippet.

    Also flags out-of-bounds spans, missing documents, non-disjoint ground-truth
    spans (which the reference metric silently double-counts) and any document
    containing CRLF line endings, where the two read modes can diverge.
    """
    root = Path(corpus_dir or CORPUS_DIR)
    problems: list[SpanProblem] = []
    overlapping: list[str] = []
    seen: set[str] = set()

    for query in queries:
        spans_by_file: dict[str, list[tuple[int, int]]] = {}
        for snippet in query.snippets:
            spans_by_file.setdefault(snippet.file_path, []).append(snippet.span)
            document = documents.get(snippet.file_path)
            if document is None:
                problems.append(
                    SpanProblem(
                        query.text,
                        snippet.file_path,
                        snippet.span,
                        "missing_document",
                        snippet.answer,
                    )
                )
                continue
            too_long = snippet.end > len(document.content)
            empty = snippet.start < 0 or snippet.start >= snippet.end
            if too_long or empty:
                problems.append(
                    SpanProblem(
                        query.text, snippet.file_path, snippet.span, "out_of_bounds", snippet.answer
                    )
                )
                continue
            actual = document.content[snippet.start : snippet.end]
            if actual != snippet.answer:
                problems.append(
                    SpanProblem(
                        query.text,
                        snippet.file_path,
                        snippet.span,
                        "mismatch",
                        snippet.answer,
                        actual,
                    )
                )
        for spans in spans_by_file.values():
            spans.sort()
            if any(spans[i - 1][1] >= spans[i][0] for i in range(1, len(spans))):
                overlapping.append(query.text)
                break

    crlf: list[str] = []
    for file_path in documents:
        seen.add(file_path)
        if b"\r\n" in (root / file_path).read_bytes():
            crlf.append(file_path)

    return VerificationReport(
        mode=mode,
        queries=len(queries),
        snippets=sum(len(q.snippets) for q in queries),
        documents=len(documents),
        problems=problems,
        crlf_documents=tuple(sorted(crlf)),
        overlapping_span_queries=tuple(overlapping),
    )
