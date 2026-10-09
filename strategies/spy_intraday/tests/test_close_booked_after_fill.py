"""A SPY daytrader close is booked when its fill is known, not when it is sent.

The ledger row used to be written as soon as the close was submitted. An urgent
market close returns from the acknowledgement, and a final limit attempt can
time out and rest, so the row went in unpriced; when that order was later
cancelled and the position closed by another order, the same position was
booked twice. Ten rows were repaired by hand on 2026-10-05 (two of them
quarantined as duplicates of a later close).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.live_4h_exec as exec_mod
from strategies.spy_intraday.Policy.order_policy import (
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)

SYMBOL = "SPY261005C00770000"
QUIET = lambda _m: None  # noqa: E731


class _Broker:
    """Answers get_order from a table the test edits as the order progresses."""

    def __init__(self):
        self.orders: dict[str, dict] = {}
        self.reads = 0

    def get_order(self, order_id):
        self.reads += 1
        order = self.orders[order_id]
        if isinstance(order, Exception):
            raise order
        return order


def _policy(broker):
    pol = OptionOrderPolicy(OptionOrderPolicyConfig(submit_orders=True))
    pol._client = broker
    pol._long_symbol = SYMBOL
    pol._long_contracts = 1
    pol._long_avg_entry_price = 1.75
    return pol


def _ledger_rows() -> list[dict]:
    path = Path(exec_mod.DEFAULT_LEDGER_ROOT) / "spy_daytrader" / "closed_trades.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _urgent_close(order_id="mkt-1", qty=1):
    """What `_submit_order_inner` returns for an urgent market close: the ack only."""
    return {"simulated": False, "intent": "close", "urgent_market": True,
            "response": {"id": order_id, "status": "pending_new", "filled_avg_price": None},
            "side": "sell", "qty": qty, "symbol": SYMBOL}


def test_an_acknowledged_close_is_booked_only_after_it_fills():
    broker = _Broker()
    pol = _policy(broker)

    pol._append_closed_trade_ledger(symbol=SYMBOL, qty=1, result=_urgent_close(), logger=QUIET)
    # The caller clears the side as soon as the close returns.
    pol._long_symbol, pol._long_contracts, pol._long_avg_entry_price = None, 0, float("nan")
    assert _ledger_rows() == []

    broker.orders["mkt-1"] = {"id": "mkt-1", "status": "accepted", "filled_qty": "0"}
    assert pol.settle_unsettled_closes(logger=QUIET) == 0
    assert _ledger_rows() == []

    broker.orders["mkt-1"] = {"id": "mkt-1", "status": "filled",
                              "filled_qty": "1", "filled_avg_price": "3.16"}
    assert pol.settle_unsettled_closes(logger=QUIET) == 1

    (row,) = _ledger_rows()
    assert row["order_id"] == "mkt-1"
    assert row["exit_fill_price"] == pytest.approx(3.16)
    assert row["entry_avg_price"] == pytest.approx(1.75)   # captured before the side was cleared
    assert row["realized_pnl"] == pytest.approx(141.0)     # (3.16 - 1.75) * 100
    assert row["option_side"] == "long"
    assert pol._unsettled_closes == []

    reads = broker.reads
    assert pol.settle_unsettled_closes(logger=QUIET) == 0   # nothing left to ask about
    assert broker.reads == reads and len(_ledger_rows()) == 1


def test_a_close_that_is_cancelled_and_retried_is_booked_once():
    """The double booking: a resting close cancelled after the grace, then closed again."""
    broker = _Broker()
    pol = _policy(broker)
    timed_out = {"simulated": False, "intent": "close", "pending_broker_reconcile": True,
                 "response": {"id": "lim-1", "status": "new"},
                 "verification": {"verified": False, "status": "new", "via": "timeout",
                                  "order_id": "lim-1", "order": {"id": "lim-1", "status": "new"}},
                 "side": "sell", "qty": 1, "symbol": SYMBOL}
    pol._append_closed_trade_ledger(symbol=SYMBOL, qty=1, result=timed_out, logger=QUIET)
    assert _ledger_rows() == []

    broker.orders["lim-1"] = {"id": "lim-1", "status": "canceled", "filled_qty": "0"}
    filled_retry = {"simulated": False, "intent": "close",
                    "response": {"id": "lim-2", "status": "pending_new"},
                    "verification": {"verified": True, "status": "filled", "via": "order_poll",
                                     "order": {"id": "lim-2", "status": "filled",
                                               "filled_qty": "1", "filled_avg_price": "1.60"}},
                    "side": "sell", "qty": 1, "symbol": SYMBOL}
    pol._append_closed_trade_ledger(symbol=SYMBOL, qty=1, result=filled_retry, logger=QUIET)

    rows = _ledger_rows()
    assert [r["order_id"] for r in rows] == ["lim-2"]
    assert rows[0]["realized_pnl"] == pytest.approx(-15.0)
    assert pol._unsettled_closes == []


def test_a_partly_filled_close_books_only_what_sold():
    broker = _Broker()
    pol = _policy(broker)
    pol._long_contracts = 20
    pol._append_closed_trade_ledger(symbol=SYMBOL, qty=20, result=_urgent_close(qty=20), logger=QUIET)

    broker.orders["mkt-1"] = {"id": "mkt-1", "status": "partially_filled",
                              "filled_qty": "19", "filled_avg_price": "0.01"}
    assert pol.settle_unsettled_closes(logger=QUIET) == 0     # still working
    broker.orders["mkt-1"] = {"id": "mkt-1", "status": "canceled",
                              "filled_qty": "19", "filled_avg_price": "0.01"}
    assert pol.settle_unsettled_closes(logger=QUIET) == 1

    (row,) = _ledger_rows()
    assert row["qty"] == 19.0
    assert row["realized_pnl"] == pytest.approx((0.01 - 1.75) * 100 * 19)


def test_a_held_close_survives_a_restart():
    broker = _Broker()
    _policy(broker)._append_closed_trade_ledger(
        symbol=SYMBOL, qty=1, result=_urgent_close(), logger=QUIET)

    broker.orders["mkt-1"] = {"id": "mkt-1", "status": "filled",
                              "filled_qty": "1", "filled_avg_price": "2.00"}
    restarted = OptionOrderPolicy(OptionOrderPolicyConfig(submit_orders=True))
    restarted._client = broker
    assert restarted.settle_unsettled_closes(logger=QUIET) == 1
    assert [r["exit_fill_price"] for r in _ledger_rows()] == [2.0]


def test_a_replayed_settlement_cannot_book_the_same_order_twice():
    """A crash after the ledger write but before the held file is rewritten."""
    broker = _Broker()
    pol = _policy(broker)
    pol._append_closed_trade_ledger(symbol=SYMBOL, qty=1, result=_urgent_close(), logger=QUIET)
    held = pol._unsettled_closes_path().read_text()
    broker.orders["mkt-1"] = {"id": "mkt-1", "status": "filled",
                              "filled_qty": "1", "filled_avg_price": "2.00"}
    assert pol.settle_unsettled_closes(logger=QUIET) == 1

    pol._unsettled_closes_path().write_text(held)     # the file as the crash left it
    restarted = OptionOrderPolicy(OptionOrderPolicyConfig(submit_orders=True))
    restarted._client = broker
    assert restarted.settle_unsettled_closes(logger=QUIET) == 0
    assert len(_ledger_rows()) == 1


def test_a_failed_broker_read_keeps_the_close_held():
    broker = _Broker()
    pol = _policy(broker)
    pol._append_closed_trade_ledger(symbol=SYMBOL, qty=1, result=_urgent_close(), logger=QUIET)

    broker.orders["mkt-1"] = OSError("timed out")
    assert pol.settle_unsettled_closes(logger=QUIET) == 0
    assert [c["order_id"] for c in pol._unsettled_closes] == ["mkt-1"]
    assert _ledger_rows() == []


def test_the_broker_maintenance_tick_settles_held_closes(monkeypatch):
    broker = _Broker()
    pol = _policy(broker)
    pol._append_closed_trade_ledger(symbol=SYMBOL, qty=1, result=_urgent_close(), logger=QUIET)
    broker.orders["mkt-1"] = {"id": "mkt-1", "status": "filled",
                              "filled_qty": "1", "filled_avg_price": "2.00"}
    # The reconcile itself is not under test: stop it right after the tick's read.
    monkeypatch.setattr(pol, "_read_broker_position_state", lambda: {"ownership_unknown": True})

    assert _ledger_rows() == []
    pol.reconcile_with_broker(logger=QUIET, force=True)

    assert [(r["order_id"], r["exit_fill_price"]) for r in _ledger_rows()] == [("mkt-1", 2.0)]
