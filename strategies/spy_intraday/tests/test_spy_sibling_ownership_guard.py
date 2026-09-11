"""The SPY daytrader must not adopt a contract another module opened.

Reproduces the 2026-09-08 cross-module claim. Intraday Structure bought
5x SPY260909P00767000 at 15:49 ET; three minutes later this policy's broker
reconcile saw an unowned SPY put in the account, adopted it as its own short,
and sold four of the five contracts. Intraday Structure then found one where it
expected five, and neither module recorded a valid P&L for the trade.

The account is shared, so "there is a SPY option in the account" has never
meant "it is mine". Swing already learned this twice and reads the shared claim
book; these tests hold the SPY path to the same rule.
"""
from __future__ import annotations

import pytest

from strategies.spy_intraday.Policy.order_policy import (
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)

INTRADAY_CONTRACT = "SPY260909P00767000"
OWN_CONTRACT = "SPY260909P00760000"


class _FakeClient:
    def __init__(self, positions):
        self._positions = positions
        self.submitted: list[dict] = []

    def get_positions(self):
        return self._positions

    def submit_order(self, **kwargs):
        self.submitted.append(kwargs)
        return {"id": "equity-1", "status": "accepted"}

    def submit_option_order(self, **kwargs):
        self.submitted.append(kwargs)
        return {"id": "opt-1", "status": "accepted"}


def _position(symbol, qty, *, asset_class="us_option", side="long", avg="1.90"):
    return {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "avg_entry_price": avg,
        "asset_class": asset_class,
    }


def _policy(positions, *, monkeypatch, owned=frozenset(), unreadable=False):
    cfg = OptionOrderPolicyConfig(submit_orders=True)
    pol = OptionOrderPolicy(cfg)
    pol._client = _FakeClient(positions)

    import core.orphan_positions as orphan

    def fake_claimed(**kwargs):
        assert "spy_daytrader" in tuple(kwargs.get("exclude", ())), (
            "the SPY book must be excluded or it would look sibling-owned"
        )
        if unreadable:
            raise orphan.ClaimBookUnreadable("intraday book is truncated")
        return set(owned)

    monkeypatch.setattr(orphan, "claimed_symbols", fake_claimed)
    return pol


def test_a_contract_another_module_owns_is_not_adopted(monkeypatch):
    """The exact 2026-09-08 position: Intraday Structure's, not the daytrader's."""

    pol = _policy(
        [_position(INTRADAY_CONTRACT, 5)],
        monkeypatch=monkeypatch,
        owned={INTRADAY_CONTRACT},
    )

    state = pol._read_broker_position_state()

    assert state["short_contracts"] == 0, "the daytrader must stay flat"
    assert state["short_symbol"] is None
    assert state["ignored_sibling_owned"] == [INTRADAY_CONTRACT]


def test_the_daytrader_still_adopts_its_own_contract(monkeypatch):
    """The guard must not blind the module to the positions it does own."""

    pol = _policy(
        [_position(OWN_CONTRACT, 3)],
        monkeypatch=monkeypatch,
        owned={INTRADAY_CONTRACT},
    )

    state = pol._read_broker_position_state()

    assert state["short_contracts"] == 3
    assert state["short_symbol"] == OWN_CONTRACT
    assert state["ignored_sibling_owned"] == []


def test_a_mixed_account_adopts_only_the_unclaimed_contract(monkeypatch):
    pol = _policy(
        [_position(INTRADAY_CONTRACT, 5), _position(OWN_CONTRACT, 3)],
        monkeypatch=monkeypatch,
        owned={INTRADAY_CONTRACT},
    )

    state = pol._read_broker_position_state()

    assert state["short_symbol"] == OWN_CONTRACT
    assert state["short_contracts"] == 3


def test_an_unreadable_claim_book_stops_the_reconcile(monkeypatch):
    """Fail closed: "unknown" must never be read as "nobody owns it".

    The lenient answer is the dangerous one — the next thing the reconcile does
    with an unowned contract is adopt it, and then manage its exit.
    """

    pol = _policy(
        [_position(INTRADAY_CONTRACT, 5)],
        monkeypatch=monkeypatch,
        unreadable=True,
    )

    state = pol._read_broker_position_state()
    assert state["ownership_unknown"] is True

    result = pol.reconcile_with_broker(logger=lambda _m: None, force=True)

    assert result.get("skipped") is True
    assert result["reason"] == "sibling_ownership_unknown"
    assert result.get("changed") is False
    assert pol._short_contracts == 0, "local state must be left alone"


def test_an_unreadable_claim_book_stops_the_startup_sync(monkeypatch):
    pol = _policy(
        [_position(INTRADAY_CONTRACT, 5)],
        monkeypatch=monkeypatch,
        unreadable=True,
    )

    result = pol.sync_from_broker(logger=lambda _m: None)

    assert result.get("skipped") is True
    assert result["reason"] == "sibling_ownership_unknown"
    assert pol._short_contracts == 0


def test_sibling_owned_underlying_shares_are_not_flattened(monkeypatch):
    """The share flatten is a market sell; it must ask who owns the shares."""

    pol = _policy(
        [_position("SPY", 200, asset_class="us_equity", avg="767.0")],
        monkeypatch=monkeypatch,
        owned={"SPY"},
    )
    assert pol.cfg.auto_flatten_underlying_shares, "the flatten must be on for this test"

    result = pol.reconcile_with_broker(logger=lambda _m: None, force=True)

    assert result.get("underlying_equity_flatten") is None
    assert pol._client.submitted == [], "no sell may be sent for someone else's shares"
