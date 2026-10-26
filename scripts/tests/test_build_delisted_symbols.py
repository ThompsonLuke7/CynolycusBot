"""The delisted-list builder only ever appends, and only symbols Alpaca no longer trades."""

from __future__ import annotations

import pytest

from core.shared_universe.delisted import parse_delisted_symbols
from scripts.build_delisted_symbols import classify, merge

pytestmark = pytest.mark.safe


def test_classify_keeps_only_symbols_missing_from_the_active_list():
    got = classify(
        ["AAPL", "holx", "ZZZZ", "NVDA"],
        active={"AAPL", "NVDA"},
        inactive={"HOLX": "Hologic Inc"},
    )
    assert got == {"HOLX": "inactive at Alpaca; Hologic Inc", "ZZZZ": "not found at Alpaca"}


def test_merge_appends_new_entries_and_keeps_existing_lines_verbatim():
    existing = "# header\nHOLX   # hand-written note\n"

    text, added = merge(existing, {"HOLX": "inactive", "QRVO": "inactive at Alpaca; last daily bar 2026-10-02"})

    assert added == ["QRVO"]
    assert text.startswith(existing)
    assert parse_delisted_symbols(text) == {"HOLX", "QRVO"}


def test_merge_is_a_no_op_when_nothing_is_new():
    existing = "HOLX\n"
    assert merge(existing, {"HOLX": "inactive"}) == (existing, [])
