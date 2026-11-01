"""Tests for the recursive, sentence and clause chunkers.

Every chunker must satisfy two properties: the span contract holds, and the chunk
spans cover the document (or the grid's precision numbers are meaningless).
"""

from __future__ import annotations

import pytest

from src.chunking import build_chunker, validate_chunks
from src.chunking.clause import ClauseChunker, heading_offsets
from src.chunking.recursive import RecursiveChunker, split_span
from src.chunking.sentence import SentenceChunker, sentence_spans
from src.data import Document

LEGAL_TEXT = (
    "EXHIBIT 10.1\n"
    "\n"
    "SERVICE AGREEMENT\n"
    "\n"
    "This Service Agreement is made on 1 April, 2020 between Theravance Biopharma UK\n"
    "Limited and Brett Haumann.\n"
    "\n"
    "1. APPOINTMENT\n"
    "\n"
    "1.1 The Company appoints the Executive as Chief Business Officer on the terms\n"
    "set out in this Agreement. The appointment shall begin on the Commencement Date.\n"
    "\n"
    "1.2 The Executive shall devote his full working time to the Company's business.\n"
    "\n"
    "2. TERMINATION\n"
    "\n"
    "2.1 Either party may terminate this Agreement for cause if the other party is in\n"
    "material breach and fails to cure within thirty (30) days of written notice.\n"
    "\n"
    "3. GOVERNING LAW\n"
    "\n"
    "3.1 This Agreement shall be governed by the laws of England and Wales.\n"
)

ABBREVIATION_TEXT = (
    "Acme Inc. entered into an agreement with Beta Ltd. on 3 May. "
    "The governing law is the law of the State of Delaware. It shall survive."
)


def doc(text: str = LEGAL_TEXT, name: str = "cuad/test.txt") -> Document:
    return Document(file_path=name, content=text)


def coverage(document: Document, chunks: list) -> float:
    """Fraction of document characters covered by at least one chunk."""
    covered = bytearray(len(document.content))
    for chunk in chunks:
        for index in range(chunk.start, chunk.end):
            covered[index] = 1
    return sum(covered) / len(document.content) if document.content else 1.0


# --------------------------------------------------------------------------- #
# Recursive
# --------------------------------------------------------------------------- #
def test_recursive_satisfies_span_contract() -> None:
    document = doc()
    for size in (120, 200, 400):
        chunks = RecursiveChunker(size=size, overlap=0).chunk(document)
        assert chunks
        validate_chunks(document, chunks)


def test_recursive_chunks_never_exceed_size() -> None:
    document = doc()
    for size in (120, 200, 400):
        chunks = RecursiveChunker(size=size, overlap=0).chunk(document)
        assert max(chunk.length for chunk in chunks) <= size


def test_recursive_covers_the_whole_document() -> None:
    document = doc()
    chunks = RecursiveChunker(size=150, overlap=0).chunk(document)
    assert coverage(document, chunks) == pytest.approx(1.0)
    assert chunks[0].start == 0 and chunks[-1].end == len(document.content)


def test_recursive_prefers_paragraph_boundaries() -> None:
    document = doc()
    chunks = RecursiveChunker(size=90, overlap=0).chunk(document)
    # At least some chunks should end right after a paragraph break.
    assert sum(1 for chunk in chunks if chunk.text.endswith("\n")) >= 3


def test_recursive_overlap_is_applied() -> None:
    document = doc("x" * 300, name="cuad/plain.txt")
    chunks = RecursiveChunker(size=100, overlap=20).chunk(document)
    assert chunks[0].start == 0
    assert [chunk.start for chunk in chunks][:3] == [0, 80, 180]
    assert chunks[1].text[:20] == chunks[0].text[-20:]


def test_recursive_handles_empty_and_tiny_documents() -> None:
    assert RecursiveChunker().chunk(doc("")) == []
    tiny = doc("short")
    assert [c.text for c in RecursiveChunker(size=100).chunk(tiny)] == ["short"]


def test_split_span_is_a_partition() -> None:
    text = "alpha beta gamma delta epsilon zeta"
    pieces = split_span(text, 0, len(text), 12)
    assert pieces[0][0] == 0 and pieces[-1][1] == len(text)
    for (_, previous_end), (next_start, _) in zip(pieces, pieces[1:], strict=False):
        assert previous_end == next_start


# --------------------------------------------------------------------------- #
# Sentence
# --------------------------------------------------------------------------- #
def test_sentence_satisfies_span_contract_and_coverage() -> None:
    document = doc()
    chunks = SentenceChunker(size=200).chunk(document)
    validate_chunks(document, chunks)
    assert coverage(document, chunks) > 0.99


def test_sentence_does_not_split_after_abbreviations() -> None:
    spans = sentence_spans(ABBREVIATION_TEXT)
    texts = [ABBREVIATION_TEXT[a:b] for a, b in spans]
    assert "Acme Inc. entered into an agreement with Beta Ltd. on 3 May. " in texts


def test_sentence_keeps_sentences_intact() -> None:
    document = doc(ABBREVIATION_TEXT, name="cuad/sent.txt")
    chunks = SentenceChunker(size=200).chunk(document)
    assert len(chunks) == 1  # short enough to be one window
    assert chunks[0].text == ABBREVIATION_TEXT


def test_sentence_windows_respect_size() -> None:
    document = doc()
    chunks = SentenceChunker(size=150).chunk(document)
    # A window may exceed `size` only when a single sentence is longer than it.
    for chunk in chunks:
        if chunk.length > 150:
            assert len(sentence_spans(chunk.text)) == 1


def test_sentence_handles_empty_document() -> None:
    assert SentenceChunker().chunk(doc("")) == []


# --------------------------------------------------------------------------- #
# Clause
# --------------------------------------------------------------------------- #
def test_clause_finds_headings() -> None:
    offsets = heading_offsets(LEGAL_TEXT)
    starts = [LEGAL_TEXT[offset : offset + 24].splitlines()[0] for offset in offsets]
    joined = " | ".join(starts)
    assert "1. APPOINTMENT" in joined
    assert "2. TERMINATION" in joined
    assert "3. GOVERNING LAW" in joined


def test_clause_satisfies_span_contract_and_coverage() -> None:
    document = doc()
    chunks = ClauseChunker(max_size=400, min_size=80).chunk(document)
    validate_chunks(document, chunks)
    assert coverage(document, chunks) == pytest.approx(1.0)


def test_clause_respects_max_size() -> None:
    document = doc()
    chunks = ClauseChunker(max_size=120, min_size=40).chunk(document)
    assert max(chunk.length for chunk in chunks) <= 120


def test_clause_merges_tiny_fragments() -> None:
    document = doc()
    chunks = ClauseChunker(max_size=1024, min_size=200).chunk(document)
    # Sections coalesce into fewer, larger chunks rather than one per heading.
    assert len(chunks) < len(heading_offsets(document.content)) + 1
    assert coverage(document, chunks) == pytest.approx(1.0)
    assert max(chunk.length for chunk in chunks) <= 1024


def test_clause_handles_documents_without_headings() -> None:
    document = doc("just a wall of text with no headings at all " * 20, name="cuad/flat.txt")
    chunks = ClauseChunker(max_size=200, min_size=50).chunk(document)
    validate_chunks(document, chunks)
    assert coverage(document, chunks) == pytest.approx(1.0)


def test_clause_handles_empty_document() -> None:
    assert ClauseChunker().chunk(doc("")) == []


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["fixed", "recursive", "sentence", "clause"])
def test_registry_builds_every_chunker(name: str) -> None:
    params = {"max_size": 512} if name == "clause" else {"size": 512}
    chunker = build_chunker(name, params)
    assert chunker.name == name
    document = doc()
    validate_chunks(document, chunker.chunk(document))
