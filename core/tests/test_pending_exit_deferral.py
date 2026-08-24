"""After-close EQUITY exit deferral: queue exits the broker's OPG window rejects.

Regression cover for 2026-08-03. equity_order_tif() returns 'opg' whenever the
market is closed, but Alpaca only accepts an OPG order between 19:00 and 09:28
ET. The 4H runners fire at ~16:20-16:35 ET, inside the 16:00-19:00 dead zone, so
Meta's CRWV take_profit_+30% sell came back:

  HTTP 403 {"code":40310000,"message":"opg orders must be submitted after
            7:00pm and before 9:28am"}

Entries already survived this via the pending-open queue; exits did not.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from core.live_4h_exec import (
    defer_exits_if_opg_unavailable,
    opg_window_is_open,
    submit_pending_exit_orders,
)

_ET = ZoneInfo("America/New_York")


def _et(hour, minute, day=3):
    return datetime(2026, 8, day, hour, minute, tzinfo=_ET).astimezone(timezone.utc)


class _Client:
    def __init__(self, *, reject=()):
        self.reject = set(reject)
        self.sent = []
        self.option_orders = []

    def submit_order(self, *, symbol, qty, side, **k):
        self.sent.append((side, symbol, qty))
        if symbol in self.reject:
            raise RuntimeError("HTTP Error 403: Forbidden")
        return {"id": f"{symbol}-{side}"}

    # Options take a different broker endpoint; the flush must not send them
    # down the equity market-order path.
    def get_option_quotes(self, symbols, **k):
        return {"quotes": {symbols: {"bp": 0.60, "ap": 0.70}}}

    def submit_option_order(self, *, symbol, qty, side, order_type, **k):
        self.option_orders.append((side, symbol, qty, order_type, k.get("limit_price")))
        if symbol in self.reject:
            raise RuntimeError("HTTP Error 422: Unprocessable Entity")
        return {"id": f"{symbol}-{side}"}


# --- opg_window_is_open --------------------------------------------------------

@pytest.mark.parametrize("hour,minute,expected", [
    (19, 0, True),    # window opens
    (23, 30, True),
    (2, 0, True),
    (9, 27, True),    # last accepted minute
    (9, 28, False),   # window closes
    (10, 0, False),
    (16, 20, False),  # Meta's actual run time — the dead zone
    (16, 35, False),  # Momentum / HTF run time
    (18, 59, False),  # one minute before it reopens
])
def test_opg_window_boundaries(hour, minute, expected):
    assert opg_window_is_open(_et(hour, minute)) is expected


# --- defer_exits_if_opg_unavailable -------------------------------------------

def test_market_open_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    plan = [("CRWV", "sell", 12, "take_profit_+30%", "equity")]
    out = defer_exits_if_opg_unavailable("meta_ranker", "bar", plan, {}, ledger_root=str(tmp_path))
    assert out == plan
    assert not (tmp_path / "meta_ranker" / "pending_exit_orders.json").exists()


def test_inside_opg_window_submits_normally(monkeypatch, tmp_path):
    """22:00 ET: market closed, but an OPG order is accepted — do not defer."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    plan = [("CRWV", "sell", 12, "take_profit_+30%", "equity")]
    out = defer_exits_if_opg_unavailable("meta_ranker", "bar", plan, {},
                                         now=_et(22, 0), ledger_root=str(tmp_path))
    assert out == plan
    assert not (tmp_path / "meta_ranker" / "pending_exit_orders.json").exists()


def test_dead_zone_queues_equity_exits(monkeypatch, tmp_path):
    """The CRWV case: 16:20 ET, market closed, OPG unavailable."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    plan = [("CRWV", "sell", 12, "take_profit_+30%", "equity")]
    out = defer_exits_if_opg_unavailable("meta_ranker", "2026-08-03 18:00", plan, {},
                                         now=_et(16, 20), ledger_root=str(tmp_path))
    assert out == []
    q = json.loads((tmp_path / "meta_ranker" / "pending_exit_orders.json").read_text())
    assert len(q["entries"]) == 1
    rec = q["entries"][0]
    assert rec["order_symbol"] == "CRWV"
    assert rec["qty"] == 12
    assert rec["reason"] == "take_profit_+30%"


def test_option_exits_are_deferred_too(monkeypatch, tmp_path):
    """2026-08-05 CLSK: option exits DO get rejected after the close.

    This test used to assert the opposite ("options never hit the OPG
    restriction"). Options are day-only on an RTH-only instrument, so a market
    sell after 16:00 returns 422 "options market orders are only allowed during
    market hours". HTF's CLSK260821C00015000 stop_-39% failed exactly that way
    at 16:25 and the call was carried overnight unstopped. Entries are still
    left alone here — they have their own pending-open queue.
    """
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    plan = [
        ("CRWV", "sell", 12, "take_profit_+30%", "equity"),          # deferred
        ("BBB260821C00050000", "sell", 10, "stop_-39%", "option"),   # deferred
        ("AAA", "buy", 100, "entry", "equity"),                      # kept (entry path)
    ]
    out = defer_exits_if_opg_unavailable("meta_ranker", "bar", plan, {},
                                         now=_et(16, 20), ledger_root=str(tmp_path))
    assert out == [("AAA", "buy", 100, "entry", "equity")]
    queued = json.loads((tmp_path / "meta_ranker" / "pending_exit_orders.json").read_text())
    assert {e["order_symbol"]: e["route"] for e in queued["entries"]} == {
        "CRWV": "equity", "BBB260821C00050000": "option",
    }


def test_option_exits_defer_even_inside_the_opg_window(monkeypatch, tmp_path):
    """The OPG window rescues equity exits; it does nothing for options.

    At 22:00 ET an equity OPG sell is accepted by the broker and should go out
    now. An option sell is still refused, so it must still be queued.
    """
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    plan = [
        ("CRWV", "sell", 12, "take_profit_+30%", "equity"),          # kept: OPG works
        ("BBB260821C00050000", "sell", 10, "stop_-39%", "option"),   # deferred
    ]
    out = defer_exits_if_opg_unavailable("meta_ranker", "bar", plan, {},
                                         now=_et(22, 0), ledger_root=str(tmp_path))
    assert out == [("CRWV", "sell", 12, "take_profit_+30%", "equity")]
    queued = json.loads((tmp_path / "meta_ranker" / "pending_exit_orders.json").read_text())
    assert [e["order_symbol"] for e in queued["entries"]] == ["BBB260821C00050000"]


def test_deferring_an_exit_never_touches_managed_state(monkeypatch, tmp_path):
    """A deferred exit means the position is STILL HELD — it must stay tracked.

    This is the one place the exit path must differ from the entry path, which
    prunes new_managed because nothing was placed.
    """
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    managed = {"CRWV": {"route": "equity", "symbol": "CRWV", "shares": 75}}
    before = json.dumps(managed, sort_keys=True)
    plan = [("CRWV", "sell", 12, "take_profit_+30%", "equity")]
    defer_exits_if_opg_unavailable("meta_ranker", "bar", plan, {},
                                   now=_et(16, 20), ledger_root=str(tmp_path))
    assert json.dumps(managed, sort_keys=True) == before


def test_requeueing_the_same_symbol_keeps_only_the_newest(monkeypatch, tmp_path):
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    for qty, reason in ((12, "take_profit_+30%"), (75, "horizon")):
        defer_exits_if_opg_unavailable("meta_ranker", "bar",
                                       [("CRWV", "sell", qty, reason, "equity")], {},
                                       now=_et(16, 20), ledger_root=str(tmp_path))
    q = json.loads((tmp_path / "meta_ranker" / "pending_exit_orders.json").read_text())
    assert len(q["entries"]) == 1
    assert q["entries"][0]["qty"] == 75
    assert q["entries"][0]["reason"] == "horizon"


def test_unknown_module_is_a_noop(tmp_path):
    plan = [("CRWV", "sell", 12, "take_profit_+30%", "equity")]
    assert defer_exits_if_opg_unavailable(None, "bar", plan, {}, ledger_root=str(tmp_path)) == plan


# --- submit_pending_exit_orders -----------------------------------------------

def _queue(tmp_path, entries):
    d = tmp_path / "meta_ranker"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pending_exit_orders.json").write_text(json.dumps({"updated": "x", "entries": entries}))


def test_flush_submits_a_still_held_exit(tmp_path):
    _queue(tmp_path, [{"order_symbol": "CRWV", "side": "sell", "qty": 12,
                       "route": "equity", "reason": "take_profit_+30%"}])
    c = _Client()
    res = submit_pending_exit_orders(c, "meta_ranker", equity_tif_fn=lambda: "day",
                                     pos_lookup={"CRWV": {"qty": 75}}, ledger_root=str(tmp_path))
    assert res["count"] == 1
    assert c.sent == [("sell", "CRWV", 12)]
    # queue drained
    q = json.loads((tmp_path / "meta_ranker" / "pending_exit_orders.json").read_text())
    assert q["entries"] == []


def test_flush_routes_an_option_exit_through_the_option_endpoint(tmp_path):
    """A queued option exit must not be sent as an equity market order.

    The contract carries a far-future expiry on purpose: this test is about
    routing, and the original CLSK260821C00015000 fixture would now be dropped
    by the expiry check before it ever reached the option endpoint.
    """
    _queue(tmp_path, [{"order_symbol": "CLSK991231C00015000", "side": "sell", "qty": 43,
                       "route": "option", "reason": "stop_-39%"}])
    c = _Client()
    res = submit_pending_exit_orders(c, "meta_ranker", equity_tif_fn=lambda: "day",
                                     pos_lookup={"CLSK991231C00015000": {"qty": 43}},
                                     ledger_root=str(tmp_path))
    assert res["count"] == 1
    assert c.sent == []  # never touched the equity path
    assert c.option_orders[0][:3] == ("sell", "CLSK991231C00015000", 43)


def test_flush_skips_a_position_already_gone(tmp_path):
    """Closed by hand or by a sibling reconcile between the queue and the flush."""
    _queue(tmp_path, [{"order_symbol": "CRWV", "side": "sell", "qty": 12, "route": "equity"}])
    c = _Client()
    res = submit_pending_exit_orders(c, "meta_ranker", equity_tif_fn=lambda: "day",
                                     pos_lookup={}, ledger_root=str(tmp_path))
    assert res["count"] == 0
    assert c.sent == []
    assert res["skipped"][0]["skip"] == "not_held"


def test_flush_clamps_qty_to_what_is_actually_held(tmp_path):
    """Never try to sell more than the broker says we own."""
    _queue(tmp_path, [{"order_symbol": "CRWV", "side": "sell", "qty": 75, "route": "equity"}])
    c = _Client()
    res = submit_pending_exit_orders(c, "meta_ranker", equity_tif_fn=lambda: "day",
                                     pos_lookup={"CRWV": {"qty": 40}}, ledger_root=str(tmp_path))
    assert res["count"] == 1
    assert c.sent == [("sell", "CRWV", 40)]


def test_flush_records_a_rejected_order_without_raising(tmp_path):
    _queue(tmp_path, [{"order_symbol": "CRWV", "side": "sell", "qty": 12, "route": "equity"}])
    c = _Client(reject={"CRWV"})
    res = submit_pending_exit_orders(c, "meta_ranker", equity_tif_fn=lambda: "day",
                                     pos_lookup={"CRWV": {"qty": 75}}, ledger_root=str(tmp_path))
    assert res["count"] == 0
    assert "submit_failed" in res["skipped"][0]["skip"]


def test_flush_with_no_queue_is_a_noop(tmp_path):
    res = submit_pending_exit_orders(_Client(), "meta_ranker", equity_tif_fn=lambda: "day",
                                     pos_lookup={}, ledger_root=str(tmp_path))
    assert res == {"submitted": [], "skipped": [], "count": 0}


# --- a failed exit stays queued (2026-08-18) ----------------------------------
#
# The flush used to end with an unconditional `entries: []` write, so an exit
# whose submit raised was erased along with the ones that went through. On
# 2026-08-18 meta_ranker's deferred INDI sell failed on a dead nervous-system
# Postgres at the 09:35 flush; the queue was cleared and 1,475 shares stayed in
# the book with no pending exit and no retry. Entries retried forever while
# exits got exactly one attempt — wrong in opposite directions.

def test_a_failed_exit_stays_queued_for_the_next_flush(tmp_path):
    client = _Client(reject={"INDI"})
    q = tmp_path / "meta_ranker"
    q.mkdir(parents=True)
    (q / "pending_exit_orders.json").write_text(json.dumps({
        "updated": "x",
        "entries": [
            {"order_symbol": "INDI", "side": "sell", "qty": 236, "route": "equity",
             "reason": "stop_-39%", "bar": "2026-08-18 18:00"},
            {"order_symbol": "OKAY", "side": "sell", "qty": 10, "route": "equity",
             "reason": "stop_-39%", "bar": "2026-08-18 18:00"},
        ],
    }))

    res = submit_pending_exit_orders(
        client, "meta_ranker", equity_tif_fn=lambda: "day",
        pos_lookup={"INDI": {"qty": 236, "avg_entry": 5.0},
                    "OKAY": {"qty": 10, "avg_entry": 5.0}},
        ledger_root=str(tmp_path),
    )

    assert res["count"] == 1, "the healthy exit still goes"
    remaining = json.loads((q / "pending_exit_orders.json").read_text())["entries"]
    assert [r["order_symbol"] for r in remaining] == ["INDI"]
    assert remaining[0]["attempts"] == 1
    assert "403" in remaining[0]["last_skip"]


def test_a_position_that_is_already_gone_is_not_requeued(tmp_path):
    """`not_held` is terminal — there is nothing left to sell, so retrying it
    forever would be the pending-open bug in the other queue."""
    client = _Client()
    q = tmp_path / "meta_ranker"
    q.mkdir(parents=True)
    (q / "pending_exit_orders.json").write_text(json.dumps({
        "updated": "x",
        "entries": [{"order_symbol": "SOLD", "side": "sell", "qty": 5,
                     "route": "equity", "reason": "stop_-39%", "bar": "b"}],
    }))

    submit_pending_exit_orders(
        client, "meta_ranker", equity_tif_fn=lambda: "day",
        pos_lookup={}, ledger_root=str(tmp_path),
    )

    assert json.loads((q / "pending_exit_orders.json").read_text())["entries"] == []


# --- expiry and attempt ceiling ------------------------------------------------

def test_flush_drops_an_expired_contract(tmp_path, caplog):
    """An expired option cannot be sold, so the queue stops carrying it.

    meta_ranker queued a 1-contract MRVL260821C00210000 take-profit on the
    2026-08-20 18:00 UTC bar. The risk pass sold all 4 contracts on 08-21 for
    expiring_before_closure, but the queue file was last written at 09:35 that
    morning and still listed the row. Absent the position it is caught by
    "not_held" — but a contract the broker still reports would otherwise be
    retried every session forever against a symbol that no longer trades.
    """
    _queue(tmp_path, [{"order_symbol": "MRVL260821C00210000", "side": "sell", "qty": 1,
                       "route": "option", "reason": "take_profit_+30%"}])
    c = _Client()
    res = submit_pending_exit_orders(
        c, "meta_ranker", equity_tif_fn=lambda: "day",
        pos_lookup={"MRVL260821C00210000": {"qty": 1}}, ledger_root=str(tmp_path),
    )

    assert res["count"] == 0
    assert c.option_orders == []
    assert res["skipped"][0]["skip"] == "contract_expired:2026-08-21"
    # Terminal: it does not come back on the next flush.
    q = json.loads((tmp_path / "meta_ranker" / "pending_exit_orders.json").read_text())
    assert q["entries"] == []


def test_flush_keeps_an_unexpired_contract(tmp_path):
    """The expiry check must not eat a live contract."""
    _queue(tmp_path, [{"order_symbol": "CLSK991231C00015000", "side": "sell", "qty": 5,
                       "route": "option", "reason": "stop_-39%"}])
    c = _Client()
    res = submit_pending_exit_orders(
        c, "meta_ranker", equity_tif_fn=lambda: "day",
        pos_lookup={"CLSK991231C00015000": {"qty": 5}}, ledger_root=str(tmp_path),
    )
    assert res["count"] == 1


def test_repeatedly_failing_exit_escalates_but_stays_queued(tmp_path, caplog):
    """meta_ranker's AMLX trim failed on 08-18, 08-20 and 08-21 with nothing
    louder than a WARNING naming the symbol, against a position sitting at
    +$3,832 that the module wanted to trim. It must keep retrying — dropping an
    open risk decision is what stranded 1,475 INDI shares on 08-18 — but it must
    stop being quiet about it.
    """
    _queue(tmp_path, [{"order_symbol": "AMLX", "side": "sell", "qty": 36, "route": "equity",
                       "reason": "take_profit_+30%", "attempts": 3,
                       "last_skip": "submit_failed: POLICY_VETO"}])
    c = _Client(reject={"AMLX"})
    with caplog.at_level("ERROR"):
        res = submit_pending_exit_orders(
            c, "meta_ranker", equity_tif_fn=lambda: "day",
            pos_lookup={"AMLX": {"qty": 227}}, ledger_root=str(tmp_path),
        )

    assert res["count"] == 0
    assert "needs a human look" in caplog.text
    # Still queued, attempt count advanced.
    q = json.loads((tmp_path / "meta_ranker" / "pending_exit_orders.json").read_text())
    assert [e["order_symbol"] for e in q["entries"]] == ["AMLX"]
    assert q["entries"][0]["attempts"] == 4
