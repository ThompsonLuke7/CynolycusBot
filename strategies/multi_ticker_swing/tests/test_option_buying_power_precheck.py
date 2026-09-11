"""Swing option entries the account cannot pay for are skipped, not submitted.

The 4H family gained this pre-filter after 2026-09-02
(`core.live_4h_exec.filter_option_entries_by_buying_power`); the 30m swing
runner never did. On 2026-09-08 a KDP261016C00033000 buy needing $5,025.03 was
submitted against $919.55 of options buying power and burned its whole
three-rung limit ladder collecting `HTTP Error 403 ... insufficient options
buying power`. The constraint appeared only in the server log.

It is a pre-filter, never a gate: anything it cannot establish must let the
order through exactly as before.
"""
from __future__ import annotations

import pytest

from strategies.multi_ticker_swing.live.runner import SwingLiveRunner


class _Client:
    def __init__(self, bp="919.55", raises=False):
        self._bp = bp
        self._raises = raises

    def get_account(self):
        if self._raises:
            raise RuntimeError("account unreadable")
        return {"options_buying_power": self._bp}


class _NoAccountClient:
    """An adapter without the endpoint at all."""


def _runner(client):
    runner = SwingLiveRunner.__new__(SwingLiveRunner)
    runner._client = client
    return runner


def test_an_unaffordable_entry_is_reported_unfunded():
    """The exact 2026-09-08 order: 67 contracts at 0.75 = $5,025 vs $919.55."""

    runner = _runner(_Client(bp="919.55"))

    out = runner._option_entry_unfunded(
        option_symbol="KDP261016C00033000", qty=67, limit_price=0.75)

    assert out is not None
    assert out["cost"] == pytest.approx(5025.0)
    assert out["available"] == pytest.approx(919.55)
    assert "insufficient options buying power" in out["reason"]


def test_an_affordable_entry_passes_through():
    runner = _runner(_Client(bp="10000"))

    assert runner._option_entry_unfunded(
        option_symbol="KDP261016C00033000", qty=1, limit_price=0.75) is None


def test_an_exactly_affordable_entry_passes_through():
    """Cost equal to the balance is fundable; the filter must not round it off."""

    runner = _runner(_Client(bp="5025.00"))

    assert runner._option_entry_unfunded(
        option_symbol="KDP261016C00033000", qty=67, limit_price=0.75) is None


@pytest.mark.parametrize("client", [_Client(raises=True), _NoAccountClient()],
                         ids=["unreadable_account", "no_account_endpoint"])
def test_an_unknown_balance_never_blocks_the_order(client):
    """A pre-filter that cannot read the account must not become a gate."""

    runner = _runner(client)

    assert runner._option_entry_unfunded(
        option_symbol="KDP261016C00033000", qty=67, limit_price=0.75) is None


@pytest.mark.parametrize("qty,limit", [(0, 0.75), (67, None), (67, 0.0)],
                         ids=["zero_qty", "no_limit_price", "zero_price"])
def test_an_unpriced_order_is_not_filtered(qty, limit):
    runner = _runner(_Client(bp="1"))

    assert runner._option_entry_unfunded(
        option_symbol="KDP261016C00033000", qty=qty, limit_price=limit) is None
