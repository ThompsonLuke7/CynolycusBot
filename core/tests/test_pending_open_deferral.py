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


def test_stale_readiness_after_close_queues_entries_instead_of_discarding(monkeypatch, tmp_path):
    """A stale stamp at ~16:20 ET must not destroy the after-close signal stream.

    Regression for 2026-07-29, when execute_plan ran the readiness gate BEFORE
    defer_entries_if_market_closed. The gate stripped every buy and popped it
    from new_managed, so the deferral step saw no reason=="entry" item and wrote
    no pending-open queue: all 28 after-close entries across Meta/HTF/Momentum
    were silently discarded rather than held for the next open. Submission is
    still gated -- submit_pending_open_entries re-runs the readiness check at
    flush time (covered by the test above).
    """
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: False)
    # Stale stamp: the real gate strips every buy and prunes new_managed.
    monkeypatch.setattr(
        ex,
        "filter_entry_orders_for_readiness",
        lambda plan, **kwargs: (
            [p for p in plan if str(p[1]).lower() != "buy"],
            {"AAA", "BBB"},
            "readiness stamp predates latest session",
        ),
    )
    # execute_plan has no ledger_root hook; redirect the queue at its source.
    monkeypatch.setattr(ex, "pending_open_path",
                        lambda module, ledger_root="Data/inference": tmp_path / module / "pending_open_entries.json")

    plan = [
        ("AAA", "buy", 100, "entry", "equity"),
        ("BBB", "buy", 50, "entry", "equity"),
        ("OLD", "sell", 100, "horizon", "equity"),
    ]
    new_managed = {
        "AAA": {"route": "equity", "symbol": "AAA", "runs_held": 0},
        "BBB": {"route": "equity", "symbol": "BBB", "runs_held": 0},
    }

    client = _Client()
    ex.execute_plan(
        client,
        plan=plan,
        new_managed=new_managed,
        limits={},
        submit=True,
        equity_tif_fn=lambda: "day",
        module="meta_ranker",
        bar="2026-07-29 18:00:00+00:00",
    )

    queue = tmp_path / "meta_ranker" / "pending_open_entries.json"
    assert queue.exists(), "after-close entries must be queued, not discarded"
    queued = {e["order_symbol"] for e in json.loads(queue.read_text())["entries"]}
    assert queued == {"AAA", "BBB"}
    # Nothing was bought after hours, and the exit still went out.
    assert ("buy", "AAA") not in client.sent and ("buy", "BBB") not in client.sent
