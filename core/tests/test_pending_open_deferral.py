"""After-close entry deferral: queue entries when the market is shut, then a
pre-open flush re-ranks against the fresh top-K and submits at the open.
"""
from __future__ import annotations

import json

import core.live_4h_exec as ex
from core.live_4h_exec import (
    defer_entries_if_market_closed,
    submit_pending_open_entries,
)


class _Client:
    def __init__(self, *, reject=()):
        self.reject = set(reject)
        self.sent = []

    def _submit(self, sym, side):
        self.sent.append((side, sym))
        if sym in self.reject:
            raise RuntimeError("HTTP Error 403: Forbidden")
        return {"id": f"{sym}-{side}"}

    def submit_option_order(self, *, symbol, qty, side, **k):
        return self._submit(symbol, side)

    def submit_order(self, *, symbol, qty, side, **k):
        return self._submit(symbol, side)


def test_market_open_is_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "is_market_open_now", lambda now=None: True, raising=False)
    # patch the name the function imports locally
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    plan = [("AAA", "buy", 100, "entry", "equity")]
    nm = {"AAA": {"qty": 100, "route": "equity"}}
    out = defer_entries_if_market_closed("momentum_expansion", "bar", plan, nm, {}, ledger_root=str(tmp_path))
    assert out == plan and "AAA" in nm  # nothing deferred while open
    assert not (tmp_path / "momentum_expansion" / "pending_open_entries.json").exists()


def test_closed_queues_entries_keeps_exits(monkeypatch, tmp_path):
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    plan = [
        ("AAA", "buy", 100, "entry", "equity"),                 # deferred
        ("BBB260717C00050000", "buy", 10, "entry", "option"),   # deferred
        ("CCC", "sell", 100, "horizon", "equity"),              # kept (exit)
    ]
    nm = {"AAA": {"route": "equity", "symbol": "AAA"},
          "BBB": {"route": "option", "occ": "BBB260717C00050000", "contracts": 10}}
    out = defer_entries_if_market_closed("meta_ranker", "2026-07-09 18:00", plan, nm, {"AAA": None},
                                         ledger_root=str(tmp_path))
    # only the exit remains in the plan to submit now
    assert out == [("CCC", "sell", 100, "horizon", "equity")]
    # deferred entries pruned from managed (nothing was placed)
    assert nm == {}
    q = json.loads((tmp_path / "meta_ranker" / "pending_open_entries.json").read_text())
    syms = {e["order_symbol"]: e for e in q["entries"]}
    assert set(syms) == {"AAA", "BBB260717C00050000"}
    assert syms["BBB260717C00050000"]["ticker"] == "BBB"  # underlying resolved via occ


def test_flush_reranks_and_submits(monkeypatch, tmp_path):
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    monkeypatch.setattr(
        ex,
        "filter_entry_orders_for_readiness",
        lambda plan, **kwargs: (list(plan), set(), "readiness stamp OK"),
    )
    plan = [
        ("KEEP", "buy", 100, "entry", "equity"),   # still top-K -> submitted
        ("DROP", "buy", 100, "entry", "equity"),   # no longer top-K -> skipped
        ("HELD", "buy", 100, "entry", "equity"),   # already held -> skipped
    ]
    nm = {"KEEP": {"route": "equity", "symbol": "KEEP", "runs_held": 0},
          "DROP": {"route": "equity", "symbol": "DROP", "runs_held": 0},
          "HELD": {"route": "equity", "symbol": "HELD", "runs_held": 0}}
    defer_entries_if_market_closed("meta_ranker", "bar", plan, nm, {}, ledger_root=str(tmp_path))

    c = _Client()
    res = submit_pending_open_entries(
        c, "meta_ranker", targets=["KEEP", "HELD"],  # DROP fell out of the top-K
        equity_tif_fn=lambda: "day",
        pos_lookup={"HELD": {"qty": 100}},           # HELD already in the account
        ledger_root=str(tmp_path),
    )
    assert res["count"] == 1
    assert res["submitted"]["KEEP"]["symbol"] == "KEEP"
    assert ("buy", "KEEP") in c.sent and ("buy", "DROP") not in c.sent and ("buy", "HELD") not in c.sent
    skips = {s["ticker"]: s["skip"] for s in res["skipped"]}
    assert skips == {"DROP": "no_longer_top_k", "HELD": "already_held"}
    # queue cleared after the flush
    assert not (tmp_path / "meta_ranker" / "pending_open_entries.json").exists()


def test_flush_blocks_all_eligible_entries_when_readiness_is_stale(monkeypatch, tmp_path):
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    plan = [
        ("KEEP", "buy", 100, "entry", "equity"),
        ("DROP", "buy", 100, "entry", "equity"),
    ]
    nm = {
        "KEEP": {"route": "equity", "symbol": "KEEP", "runs_held": 0},
        "DROP": {"route": "equity", "symbol": "DROP", "runs_held": 0},
    }
    defer_entries_if_market_closed("meta_ranker", "bar", plan, nm, {}, ledger_root=str(tmp_path))
    monkeypatch.setattr(
        ex,
        "filter_entry_orders_for_readiness",
        lambda plan, **kwargs: ([], {"KEEP"}, "readiness stamp predates latest session"),
    )

    c = _Client()
    res = submit_pending_open_entries(
        c,
        "meta_ranker",
        targets=["KEEP"],
        equity_tif_fn=lambda: "day",
        ledger_root=str(tmp_path),
    )

    assert res["count"] == 0
    assert res["submitted"] == {}
    assert c.sent == []
    skips = {s["ticker"]: s["skip"] for s in res["skipped"]}
    assert skips["DROP"] == "no_longer_top_k"
    assert skips["KEEP"].startswith("readiness:")
    assert not (tmp_path / "meta_ranker" / "pending_open_entries.json").exists()


def test_flush_empty_queue_is_safe(tmp_path):
    res = submit_pending_open_entries(_Client(), "htf", targets=["X"],
                                      equity_tif_fn=lambda: "day", ledger_root=str(tmp_path))
    assert res == {"submitted": {}, "skipped": [], "count": 0}
