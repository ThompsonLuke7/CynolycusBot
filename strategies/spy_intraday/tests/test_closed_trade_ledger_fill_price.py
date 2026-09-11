"""A closed SPY trade must record the price it actually filled at.

On 2026-09-08 both of the daytrader's closes landed in `closed_trades.jsonl`
with `exit_fill_price: null` and `realized_pnl: null`, even though the session
log showed `ORDER VERIFIED intent=close status=filled via=order_poll` for each.
The module could not be scored on P&L at all.

The cause is that a submission response is an acknowledgement: on this path it
returns `pending_new`/`new` and carries no fill price. The fill is established
moments later by the verification poll, and the ledger was reading only the
acknowledgement.
"""
from __future__ import annotations

import json

import pytest

from strategies.spy_intraday.Policy.order_policy import (
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)

SYMBOL = "SPY260908C00768000"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Capture ledger rows instead of writing to the repository's data dir."""

    written: list[dict] = []
    import core.live_4h_exec as exec_mod

    monkeypatch.setattr(
        exec_mod, "append_closed_trade",
        lambda module, record: written.append(record),
    )
    return written


def _policy():
    cfg = OptionOrderPolicyConfig(submit_orders=True)
    pol = OptionOrderPolicy(cfg)
    pol._long_symbol = SYMBOL
    pol._long_contracts = 1
    pol._long_avg_entry_price = 0.74
    return pol


def test_the_verified_fill_price_is_recorded(ledger):
    """The acknowledgement has no price; the verified order does."""

    pol = _policy()
    result = {
        "simulated": False,
        "intent": "close",
        # Exactly what Alpaca returns on acknowledgement.
        "response": {"id": "df47cddd", "status": "pending_new",
                     "filled_avg_price": None},
        "verification": {
            "verified": True, "status": "filled", "via": "order_poll",
            "order": {"id": "df47cddd", "status": "filled",
                      "filled_avg_price": "0.48", "filled_qty": "1"},
        },
        "side": "long", "qty": 1, "symbol": SYMBOL,
    }

    pol._append_closed_trade_ledger(
        symbol=SYMBOL, qty=1, result=result, logger=lambda _m: None)

    assert len(ledger) == 1
    row = ledger[0]
    assert row["exit_fill_price"] == pytest.approx(0.48)
    assert row["entry_avg_price"] == pytest.approx(0.74)
    # (0.48 - 0.74) * 100 * 1
    assert row["realized_pnl"] == pytest.approx(-26.0)


def test_the_response_price_is_still_used_when_it_has_one(ledger):
    """A fill that arrives on the acknowledgement must still be honoured."""

    pol = _policy()
    result = {
        "simulated": False,
        "intent": "close",
        "response": {"id": "abc", "status": "filled", "filled_avg_price": "1.10"},
        "side": "long", "qty": 1, "symbol": SYMBOL,
    }

    pol._append_closed_trade_ledger(
        symbol=SYMBOL, qty=1, result=result, logger=lambda _m: None)

    assert ledger[0]["exit_fill_price"] == pytest.approx(1.10)
    assert ledger[0]["realized_pnl"] == pytest.approx(36.0)


def test_a_genuinely_unfilled_close_still_records_null(ledger):
    """No invented price when neither source knows one."""

    pol = _policy()
    result = {
        "simulated": False,
        "intent": "close",
        "response": {"id": "abc", "status": "new", "filled_avg_price": None},
        "verification": {"verified": False, "status": "new",
                         "order": {"id": "abc", "status": "new",
                                   "filled_avg_price": None}},
        "side": "long", "qty": 1, "symbol": SYMBOL,
    }

    pol._append_closed_trade_ledger(
        symbol=SYMBOL, qty=1, result=result, logger=lambda _m: None)

    assert ledger[0]["exit_fill_price"] is None
    assert ledger[0]["realized_pnl"] is None
