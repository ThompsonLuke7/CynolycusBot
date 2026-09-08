"""Option entries the account cannot pay for are skipped, not submitted.

On 2026-09-02 eight option entries came back 403 `insufficient options buying
power`: three from Momentum Expansion at 14:32 ET against ~$4,235 available
(DELL 500 call, GTLB 50 call, SRPT 22.5 call) and five from Dealer Ranker at
15:30 ET against ~$1,255 (SKM, DINO, CNH, GFS, EIX) — Momentum's accepted
entries having spent the difference in between. Nothing checked before
submitting, so the constraint appeared only as a broker rejection in the server
log and never in the module's own plan audit.
"""

from __future__ import annotations

import pytest

from core.live_4h_exec import (
    _option_cost_basis,
    filter_option_entries_by_buying_power,
)


class _Account:
    def __init__(self, options_buying_power, raises=False):
        self._bp = options_buying_power
        self._raises = raises

    def get_account(self):
        if self._raises:
            raise RuntimeError("account unreadable")
        return {"options_buying_power": str(self._bp)}


def _buy(symbol, qty):
    return (symbol, "buy", qty, "entry", "option")


def test_the_cost_basis_matches_what_the_broker_charges() -> None:
    """Alpaca's own 403 payloads on 2026-09-02 quoted these figures."""

    assert _option_cost_basis(3, 16.75) == pytest.approx(5025.0)   # DELL
    assert _option_cost_basis(20, 2.44) == pytest.approx(4880.0)   # GTLB
    assert _option_cost_basis(27, 1.87) == pytest.approx(5049.0)   # SRPT


def test_an_unaffordable_option_entry_is_skipped_before_submission() -> None:
    plan = [_buy("DELL260918C00500000", 3)]
    limits = {"DELL260918C00500000": 16.75}

    kept, skipped = filter_option_entries_by_buying_power(
        _Account(4239.19), plan, limits,
    )

    assert kept == []
    assert skipped == ["DELL260918C00500000"]


def test_the_budget_is_consumed_in_rank_order() -> None:
    """The names the module ranked highest get the remaining capacity."""

    plan = [_buy("FIRST", 3), _buy("SECOND", 3), _buy("THIRD", 3)]
    limits = {"FIRST": 10.0, "SECOND": 10.0, "THIRD": 10.0}  # $3,000 each

    kept, skipped = filter_option_entries_by_buying_power(
        _Account(6500.0), plan, limits,
    )

    assert [row[0] for row in kept] == ["FIRST", "SECOND"]
    assert skipped == ["THIRD"]


def test_equity_entries_and_exits_are_never_filtered() -> None:
    """Only an opening option order draws on options buying power."""

    plan = [
        ("AAPL", "buy", 100, "entry", "equity"),
        ("MSFT260918C00500000", "sell", 5, "exit", "option"),
        _buy("NVDA260918C00200000", 50),
    ]
    limits = {"NVDA260918C00200000": 10.0, "MSFT260918C00500000": 4.0}

    kept, skipped = filter_option_entries_by_buying_power(_Account(1000.0), plan, limits)

    assert [row[0] for row in kept] == ["AAPL", "MSFT260918C00500000"]
    assert skipped == ["NVDA260918C00200000"]


def test_an_unpriced_option_entry_is_left_alone() -> None:
    """With no limit there is no cost to compare; the broker decides."""

    plan = [_buy("XYZ260918C00010000", 5)]

    kept, skipped = filter_option_entries_by_buying_power(_Account(1.0), plan, {})

    assert kept == plan
    assert skipped == []


def test_an_unreadable_account_submits_the_plan_unfiltered() -> None:
    """A pre-filter that saves a doomed round trip, never a gate that blocks one."""

    plan = [_buy("DELL260918C00500000", 3)]
    limits = {"DELL260918C00500000": 16.75}

    kept, skipped = filter_option_entries_by_buying_power(
        _Account(0, raises=True), plan, limits,
    )

    assert kept == plan
    assert skipped == []


def test_a_client_without_an_account_endpoint_is_tolerated() -> None:
    plan = [_buy("DELL260918C00500000", 3)]

    kept, skipped = filter_option_entries_by_buying_power(object(), plan, {"DELL260918C00500000": 16.75})

    assert kept == plan
    assert skipped == []


def test_the_skip_is_recorded_in_the_plan_audit() -> None:
    """The whole point: the constraint becomes visible where decisions are read."""

    disp: dict[str, str] = {}
    managed = {"DELL": {"route": "option", "occ": "DELL260918C00500000", "contracts": 3}}

    filter_option_entries_by_buying_power(
        _Account(100.0), [_buy("DELL260918C00500000", 3)],
        {"DELL260918C00500000": 16.75}, disp=disp, new_managed=managed,
    )

    assert disp["DELL260918C00500000"] == "insufficient_options_buying_power"
    assert "DELL" not in managed, "an entry that was never sent must not stay claimed"
