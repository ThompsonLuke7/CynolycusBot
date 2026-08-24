"""The entry ladder's top rung is configurable, and defaults to the ask.

Background. Across 490 multi-rung ladders in UI/swing_audit the mean fill sits
0.326 of the way from mid to ask, for $22,025 of cumulative fill-versus-mid
slippage — $10,749 of it in August alone. _entry_limit_ladder's docstring used
to argue the mid is simply not fillable, from 24 entries over three days in
which zero filled at the mid rung; the full history says 22.0% fill at the mid
rung and 1.0% at the ask.

Capping below the ask is therefore worth having available. It is NOT turned on
here: whether a rung that rests short of the ask fills in a live market cannot
be established from Alpaca paper fills, and the ladder already misses more often
than it fills, so a change that only improves price is not obviously a win.
These tests pin that the default is a no-op and that the lever works.
"""
from __future__ import annotations

import importlib

import pytest

import core.live_4h_exec as exec_mod


def test_default_tops_out_at_the_ask():
    assert exec_mod._ENTRY_LADDER_MAX_ASK_FRACTION == 1.0
    assert exec_mod._entry_limit_ladder(1.00, 1.30) == [1.0, 1.15, 1.3]


def test_default_is_unchanged_for_a_wide_contract():
    # SPGI on 2026-08-21: mid 8.00, ask 8.52. The live ladder was
    # [8.00, 8.26, 8.52] and must stay that way at the default.
    assert exec_mod._entry_limit_ladder(8.00, 8.52) == [8.0, 8.26, 8.52]


@pytest.mark.parametrize("fraction,expected", [
    (0.5, [8.0, 8.13, 8.26]),      # half the spread
    (0.0, [8.0]),                  # mid only
])
def test_a_lower_cap_stops_short_of_the_ask(monkeypatch, fraction, expected):
    monkeypatch.setattr(exec_mod, "_ENTRY_LADDER_MAX_ASK_FRACTION", fraction)
    assert exec_mod._entry_limit_ladder(8.00, 8.52) == expected


def test_an_out_of_range_cap_is_clamped_not_obeyed(monkeypatch):
    """A fat-fingered override must never bid ABOVE the ask."""
    monkeypatch.setattr(exec_mod, "_ENTRY_LADDER_MAX_ASK_FRACTION", 3.0)
    ladder = exec_mod._entry_limit_ladder(8.00, 8.52)
    assert max(ladder) == 8.52

    monkeypatch.setattr(exec_mod, "_ENTRY_LADDER_MAX_ASK_FRACTION", -1.0)
    assert exec_mod._entry_limit_ladder(8.00, 8.52) == [8.0]


def test_a_missing_quote_still_returns_nothing():
    assert exec_mod._entry_limit_ladder(1.0, None) == []
    assert exec_mod._entry_limit_ladder(1.0, 0.0) == []


def test_swing_runner_defaults_match_todays_behaviour():
    runner = importlib.import_module("strategies.multi_ticker_swing.live.runner")
    assert runner._ENTRY_ORDER_VERIFY_TIMEOUT_SECS == 5.0
    assert runner._ENTRY_LADDER_MAX_ASK_FRACTION == 1.0
