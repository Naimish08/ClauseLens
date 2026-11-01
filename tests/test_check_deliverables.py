"""Tests for the pre-submission checklist helper."""

from __future__ import annotations

from scripts.check_deliverables import Check


def test_check_records_failures(capsys) -> None:
    check = Check()
    check.require(True, "present")
    check.require(False, "missing")
    assert check.failures == ["missing"]
    assert "FAIL" in capsys.readouterr().out


def test_check_records_warnings(capsys) -> None:
    check = Check()
    check.warn(False, "not run yet")
    assert check.warnings == ["not run yet"]
    assert "warn" in capsys.readouterr().out


def test_check_passes_cleanly(capsys) -> None:
    check = Check()
    check.require(True, "a")
    check.warn(True, "b")
    assert not check.failures and not check.warnings
