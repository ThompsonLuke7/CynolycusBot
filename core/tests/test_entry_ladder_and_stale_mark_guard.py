"""Option ENTRY laddering, and the stale-mark second opinion on option exits.

Two 2026-08 findings, both measured rather than assumed:

* Entries were a single limit AT the ask, so every fill crossed the whole
  spread. Mean fill across 22 filled 4H option entries was +7.3% over mid, and
  +12.7% for Dealer Ranker against its +20% take-profit target.
* An option exit is decided on the broker's position mark, which after the close
  is a stale or one-sided quote. AAOI marked 4.70 on a 9.20 basis at 16:30 ET,
  tripped `stop_-39%`, and traded 9.90 the next morning.
"""
from __future__ import annotations

import pytest

from core.live_4h_exec import (
    ExecPolicy,
    _entry_limit_ladder,
    build_mixed_plan,
    execute_plan,
    submit_option_entry_with_ladder,
)


# --------------------------------------------------------------------------- #
# entry limit ladder
# --------------------------------------------------------------------------- #

def test_ladder_walks_mid_to_ask():
    assert _entry_limit_ladder(1.00, 2.00, attempts=3) == [1.00, 1.50, 2.00]


def test_ladder_last_rung_is_always_the_ask():
    for mid, ask in ((0.18, 0.21), (44.86, 47.43), (1.33, 1.53)):
        assert _entry_limit_ladder(mid, ask, attempts=3)[-1] == round(ask, 2)


def test_ladder_collapses_when_mid_equals_ask():
    assert _entry_limit_ladder(2.00, 2.00, attempts=3) == [2.00]


def test_ladder_without_a_usable_mid_is_just_the_ask():
    assert _entry_limit_ladder(None, 1.50, attempts=3) == [1.50]
    assert _entry_limit_ladder(0.0, 1.50, attempts=3) == [1.50]
    # A mid above the ask is a broken quote, not a cheaper entry.
    assert _entry_limit_ladder(3.00, 1.50, attempts=3) == [1.50]


def test_ladder_is_empty_without_an_ask():
    assert _entry_limit_ladder(1.00, None) == []
    assert _entry_limit_ladder(1.00, 0.0) == []


class _LadderClient:
    """Fills at the rung named by `fills_at_rung` (1-indexed); 0 never fills."""

    def __init__(self, *, bid=1.00, mid=1.50, fills_at_rung=1):
        self.bid, self.mid = bid, mid
        self.fills_at_rung = fills_at_rung
        self.submitted: list[float] = []
        self.cancelled: list[str] = []

    def get_option_quotes(self, symbols=None, **_k):
        ask = self.mid * 2 - self.bid
        return {"quotes": {symbols: {"bp": self.bid, "ap": ask}}}

    def submit_option_order(self, *, symbol, qty, side, limit_price=None, **_k):
        self.submitted.append(limit_price)
        n = len(self.submitted)
        status = "filled" if n == self.fills_at_rung else "new"
        return {"id": f"o{n}", "status": status}

    def get_order(self, oid):
        n = int(str(oid)[1:])
        return {"id": oid, "status": "filled" if n == self.fills_at_rung else "new"}

    def cancel_order(self, oid):
        self.cancelled.append(str(oid))

    def filled(self, resp):
        """Injected poll — same verdict as `_poll_order_filled`, without its sleeps."""
        return str((resp or {}).get("status", "")).lower() == "filled"


def test_entry_ladder_stops_at_the_first_filling_rung():
    c = _LadderClient(bid=1.00, mid=1.50, fills_at_rung=1)   # ask 2.00
    resp = submit_option_entry_with_ladder(c, symbol="X", qty=3, ask=2.00,
                                           sleep_fn=lambda _s: None, poll_fn=c.filled)
    assert resp["status"] == "filled"
    assert c.submitted == [1.50]        # never escalated past the mid
    assert c.cancelled == []


def test_entry_ladder_cancels_each_unfilled_rung_before_escalating():
    """Two live buys for one contract would double the position."""
    c = _LadderClient(bid=1.00, mid=1.50, fills_at_rung=3)
    submit_option_entry_with_ladder(c, symbol="X", qty=3, ask=2.00,
                                    sleep_fn=lambda _s: None, poll_fn=c.filled)
    assert c.submitted == [1.50, 1.75, 2.00]
    assert c.cancelled == ["o1", "o2"]   # the filling rung is not cancelled


def test_entry_ladder_leaves_last_rung_resting_when_nothing_fills():
    c = _LadderClient(bid=1.00, mid=1.50, fills_at_rung=0)
    resp = submit_option_entry_with_ladder(c, symbol="X", qty=3, ask=2.00,
                                           sleep_fn=lambda _s: None, poll_fn=c.filled)
    assert c.submitted == [1.50, 1.75, 2.00]
    assert c.cancelled == ["o1", "o2"]   # the final rung stays live
    assert resp["status"] != "filled"    # caller flags the entry unconfirmed


def test_execute_plan_uses_single_shot_ask_by_default(monkeypatch):
    """Meta/Momentum/HTF must be untouched until explicitly opted in."""
    monkeypatch.setattr("core.live_4h_exec.filter_entry_orders_for_readiness",
                        lambda plan, new_managed=None: (plan, [], ""))
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    monkeypatch.setattr("core.live_4h_exec._reverify_buys_not_held",
                        lambda client, plan, nm: plan)
    c = _LadderClient(fills_at_rung=1)
    execute_plan(c, plan=[("X", "buy", 3, "entry", "option")], limits={"X": 2.00},
                 submit=True, equity_tif_fn=lambda: "day", new_managed={})
    assert c.submitted == [2.00]         # straight to the ask, one attempt


def test_execute_plan_ladders_when_opted_in(monkeypatch):
    monkeypatch.setattr("core.live_4h_exec.filter_entry_orders_for_readiness",
                        lambda plan, new_managed=None: (plan, [], ""))
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    monkeypatch.setattr("core.live_4h_exec._reverify_buys_not_held",
                        lambda client, plan, nm: plan)
    c = _LadderClient(bid=1.00, mid=1.50, fills_at_rung=1)
    execute_plan(c, plan=[("X", "buy", 3, "entry", "option")], limits={"X": 2.00},
                 submit=True, equity_tif_fn=lambda: "day", new_managed={},
                 entry_ladder=True)
    assert c.submitted == [1.50]


def test_entry_ladder_never_applies_to_sells(monkeypatch):
    """Sells keep the descending exit ladder; the two must not cross."""
    monkeypatch.setattr("core.live_4h_exec.filter_entry_orders_for_readiness",
                        lambda plan, new_managed=None: (plan, [], ""))
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    monkeypatch.setattr("core.live_4h_exec._reverify_buys_not_held",
                        lambda client, plan, nm: plan)
    c = _LadderClient(fills_at_rung=1)
    execute_plan(c, plan=[("X", "sell", 3, "stop_-39%", "option")], limits={"X": 2.00},
                 submit=True, equity_tif_fn=lambda: "day", new_managed={},
                 entry_ladder=True)
    assert c.submitted == [2.00]


# --------------------------------------------------------------------------- #
# stale option mark on exit decisions
# --------------------------------------------------------------------------- #

class _QuoteClient:
    """Serves one live two-sided quote for the corroboration check."""

    def __init__(self, bid, ask):
        self.bid, self.ask = bid, ask
        self.quote_calls = 0

    def get_option_quotes(self, symbols=None, **_k):
        self.quote_calls += 1
        return {"quotes": {symbols: {"bp": self.bid, "ap": self.ask}}}


def _managed_option(entry_bar="2026-08-12 18:00:00+00:00"):
    return {"AAOI": {"route": "option", "occ": "AAOI260821C00140000", "contracts": 5,
                     "runs_held": 1, "bars_out": 0, "trimmed": False,
                     "entry_bar": entry_bar}}


def _plan(client, managed, pos_info, targets=()):
    return build_mixed_plan(
        client, targets=list(targets), managed=managed, pos_info=pos_info,
        bar="2026-08-13 18:00:00+00:00", signal_audits={}, policy=ExecPolicy(),
        route_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no entry")),
        ref_price_fn=lambda t: 10.0, verbose=False,
    )


def test_stale_mark_cancels_a_stop_that_should_not_fire():
    """The AAOI case: broker says -48.9%, the live market says +7.6%."""
    client = _QuoteClient(bid=9.80, ask=10.00)          # mid 9.90
    res = _plan(client, _managed_option(),
                {"AAOI260821C00140000": {"qty": 5, "avg_entry": 9.20, "current": 4.70}})

    assert res.plan == []                               # the stop did NOT fire
    state = res.new_managed["AAOI"]
    assert state["last_mark_price"] == 9.90             # corroborated mark persisted
    assert state["unrealized_gain"] == pytest.approx(0.07608, abs=1e-4)
    anomaly = res.anomalies["AAOI"]
    assert anomaly["broker_mark"] == 4.70 and anomaly["quote_mid"] == 9.90
    assert anomaly["was"] == "stop_-39%" and anomaly["now"] == "hold"


def test_a_real_loss_still_stops_out():
    """Corroboration must not become a way to never take a loss."""
    client = _QuoteClient(bid=4.60, ask=4.80)           # mid 4.70, agrees with broker
    res = _plan(client, _managed_option(),
                {"AAOI260821C00140000": {"qty": 5, "avg_entry": 9.20, "current": 4.70}})

    assert [p[3] for p in res.plan] == ["stop_-39%"]
    assert "AAOI" not in res.anomalies
    assert client.quote_calls == 1                      # checked once, then traded


def test_marks_within_tolerance_are_left_alone():
    client = _QuoteClient(bid=5.10, ask=5.30)           # mid 5.20 vs broker 4.70 = 9.6%
    res = _plan(client, _managed_option(),
                {"AAOI260821C00140000": {"qty": 5, "avg_entry": 9.20, "current": 4.70}})
    assert [p[3] for p in res.plan] == ["stop_-39%"]
    assert "AAOI" not in res.anomalies


def test_no_quote_falls_back_to_the_broker_mark():
    """A quote outage must not strand a position that needs to exit."""
    class _Dead:
        def get_option_quotes(self, **_k):
            raise RuntimeError("quote service down")

    res = _plan(_Dead(), _managed_option(),
                {"AAOI260821C00140000": {"qty": 5, "avg_entry": 9.20, "current": 4.70}})
    assert [p[3] for p in res.plan] == ["stop_-39%"]


def test_held_positions_are_not_quoted():
    """The check costs one call per position TRADING, not per position held."""
    client = _QuoteClient(bid=9.80, ask=10.00)
    res = _plan(client, _managed_option(),
                {"AAOI260821C00140000": {"qty": 5, "avg_entry": 9.20, "current": 9.30}})
    assert res.plan == []
    assert client.quote_calls == 0


def test_equity_exits_do_not_call_the_option_quote_path():
    class _Boom:
        def get_option_quotes(self, **_k):
            raise AssertionError("equities must not hit the option quote path")

    managed = {"AAA": {"route": "equity", "symbol": "AAA", "shares": 100,
                       "runs_held": 1, "bars_out": 0, "trimmed": False,
                       "entry_bar": "2026-08-12 18:00:00+00:00"}}
    res = _plan(_Boom(), managed, {"AAA": {"qty": 100, "avg_entry": 10.0, "current": 5.0}})
    assert [p[3] for p in res.plan] == ["stop_-39%"]
