"""Tests for chunkers. The span contract is the point of this file."""

from __future__ import annotations

import pytest

from src.chunking import build_chunker, validate_chunks
from src.chunking.fixed import FixedChunker
from src.data import Document
from src.types import make_chunk

PARAGRAPHS = (
    "SECTION 1. DEFINITIONS\n"
    "\n"
    "1.1 \"Agreement\" means this agreement and all schedules hereto.\n"
    "1.2 \"Confidential Information\" means any non-public information disclosed.\n"
    "\n"
    "SECTION 2. GOVERNING LAW\n"
    "\n"
    "2.1 This Agreement shall be governed by the laws of the State of Delaware.\n"
)


def doc(text: str = PARAGRAPHS, name: str = "cuad/test.txt") -> Document:
    return Document(file_path=name, content=text)


def test_span_contract_holds_for_every_chunk() -> None:
    document = doc()
    chunks = FixedChunker(size=64, overlap=0, snap=False).chunk(document)
    assert chunks
    validate_chunks(document, chunks)


def test_span_contract_holds_with_snapping_and_overlap() -> None:
    document = doc()
    for size, overlap, snap in [(64, 0, True), (80, 16, True), (120, 40, False)]:
        chunks = FixedChunker(size=size, overlap=overlap, snap=snap).chunk(document)
        validate_chunks(document, chunks)


def test_coverage_is_complete_without_overlap() -> None:
    document = doc()
    chunks = FixedChunker(size=40, overlap=0, snap=False).chunk(document)
    assert "".join(chunk.text for chunk in chunks) == document.content


def test_overlap_is_exact_when_snapping_is_off() -> None:
    document = doc("x" * 250, name="cuad/plain.txt")
    chunks = FixedChunker(size=100, overlap=50, snap=False).chunk(document)
    assert [chunk.start for chunk in chunks] == [0, 50, 100, 150]
    assert chunks[-1].end == 250
    assert chunks[0].text[50:] == chunks[1].text[:50]
    assert chunks[0].end == chunks[1].start + 50


def test_snapping_ends_chunks_on_whitespace() -> None:
    document = doc("alpha beta gamma delta epsilon zeta eta theta", name="cuad/words.txt")
    chunks = FixedChunker(size=20, overlap=0, snap=True, snap_tolerance=10).chunk(document)
    for chunk in chunks[:-1]:
        assert chunk.text.endswith(" ")


def test_empty_document_produces_no_chunks() -> None:
    assert FixedChunker().chunk(doc("")) == []


def test_no_trailing_newline_does_not_lose_text() -> None:
    document = doc("no trailing newline here")
    chunks = FixedChunker(size=8, overlap=0, snap=False).chunk(document)
    assert "".join(chunk.text for chunk in chunks) == document.content


def test_progress_is_guaranteed_with_large_overlap() -> None:
    document = doc("abcdefghijklmnopqrstuvwxyz" * 3)
    chunks = FixedChunker(size=10, overlap=9, snap=False).chunk(document)
    assert len(chunks) < len(document.content) + 2
    assert all(chunk.length > 0 for chunk in chunks)


def test_determinism() -> None:
    first = FixedChunker(size=64, overlap=8, snap=True).chunk(doc())
    second = FixedChunker(size=64, overlap=8, snap=True).chunk(doc())
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        FixedChunker(size=0)
    with pytest.raises(ValueError):
        FixedChunker(size=100, overlap=-1)
    with pytest.raises(ValueError):
        FixedChunker(size=100, overlap=100)
    with pytest.raises(ValueError):
        FixedChunker(size=100, snap_tolerance=-1)


def test_registry_builds_by_config_name() -> None:
    chunker = build_chunker("fixed", {"size": 32, "overlap": 0})
    assert chunker.name == "fixed"
    with pytest.raises(ValueError, match="unknown chunker"):
        build_chunker("clause", {})


def test_validate_chunks_catches_a_corrupted_span() -> None:
    document = doc("abcdefghij")
    bad = [make_chunk(document.file_path, 0, 5, "WRONG")]
    with pytest.raises(ValueError, match="span contract violated"):
        validate_chunks(document, bad)


def test_validate_chunks_catches_out_of_bounds() -> None:
    document = doc("abc")
    bad = [make_chunk(document.file_path, 0, 99, "abc")]
    with pytest.raises(ValueError, match="out of bounds"):
        validate_chunks(document, bad)


def test_validate_chunks_catches_wrong_document() -> None:
    document = doc("abc")
    bad = [make_chunk("cuad/other.txt", 0, 3, "abc")]
    with pytest.raises(ValueError, match="belongs to"):
        validate_chunks(document, bad)
