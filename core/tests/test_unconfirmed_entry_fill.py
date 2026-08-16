"""An ACCEPTED entry order is not a position. Label it until the broker agrees.

2026-08-11: Dealer Ranker submitted nine limit entries at 15:53 ET, seven minutes
before the close. Four (UMC 19C, P 110C, GRAB 3.5C, TSCO 36C) were accepted and
never filled, but build_mixed_plan writes a new position into managed when it
PLANS the entry and execute_plan only removes it if the submission raises — so
`live_state.json` claimed 11 managed positions against the broker's 7.

The flag is deliberately not a fill poll and never drops the entry: a limit that
fills later would become an unowned position, which is how Swing force-sold
Dealer Ranker's IOT260724C00031500 for -$4,945 on 2026-07-23. Staying claimed is
the conservative side for sibling reconciliation; build_mixed_plan's existing
qty<=0 drop settles it on the next pass either way.
"""
from __future__ import annotations

from core.live_4h_exec import ExecPolicy, build_mixed_plan, execute_plan

_UMC = "UMC260821C00019000"


class _Client:
    def __init__(self, reject=()):
        self.reject = set(reject)

    def submit_option_order(self, *, symbol, qty, side, **k):
        if symbol in self.reject:
            raise RuntimeError("HTTP Error 422")
        return {"id": f"{symbol}-oid"}

    def submit_order(self, *, symbol, qty, side, **k):
        if symbol in self.reject:
            raise RuntimeError("HTTP Error 403")
        return {"id": f"{symbol}-oid"}

    def get_positions(self):
        return []


def _exec(client, plan, new_managed, **kw):
    return execute_plan(client, plan=plan, limits={_UMC: 0.68}, submit=True,
                        equity_tif_fn=lambda: "day", new_managed=new_managed,
                        module="dealer_ranker", pos_lookup={}, bar="2026-08-11 19:53",
                        **kw)


def test_accepted_entry_is_flagged_unconfirmed(monkeypatch):
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    new_managed = {"UMC": {"route": "option", "occ": _UMC, "contracts": 57}}
    _exec(_Client(), [(_UMC, "buy", 57, "entry", "option")], new_managed)
    assert new_managed["UMC"]["pending_fill"] is True
    assert new_managed["UMC"]["entry_order_id"] == f"{_UMC}-oid"


def test_unconfirmed_entry_is_still_claimed(monkeypatch):
    """Must NOT be dropped — a late fill would otherwise be an unowned position."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    new_managed = {"UMC": {"route": "option", "occ": _UMC, "contracts": 57}}
    _exec(_Client(), [(_UMC, "buy", 57, "entry", "option")], new_managed)
    assert "UMC" in new_managed


def test_rejected_entry_is_dropped_not_flagged(monkeypatch):
    """The existing drop_failed_entry path still owns outright rejections."""
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    new_managed = {"UMC": {"route": "option", "occ": _UMC, "contracts": 57}}
    failed = _exec(_Client(reject={_UMC}), [(_UMC, "buy", 57, "entry", "option")], new_managed)
    assert failed == {_UMC}
    assert new_managed == {}


def test_a_sell_is_never_flagged(monkeypatch):
    monkeypatch.setattr("core.calendar.is_market_open_now", lambda now=None: True)
    new_managed = {"UMC": {"route": "option", "occ": _UMC, "contracts": 57}}
    _exec(_Client(), [(_UMC, "sell", 57, "stop_-50%", "option")], new_managed,
          exit_context={_UMC: ("UMC", {"route": "option", "occ": _UMC})})
    assert "pending_fill" not in new_managed.get("UMC", {})


def test_next_pass_clears_the_flag_once_the_broker_confirms():
    """build_mixed_plan sees qty>0 and the entry has settled."""
    managed = {"UMC": {"route": "option", "occ": _UMC, "contracts": 57,
                       "pending_fill": True, "entry_order_id": "x", "runs_held": 0}}
    res = build_mixed_plan(
        None, targets=["UMC"], managed=managed,
        pos_info={_UMC: {"qty": 57, "avg_entry": 0.68, "current": 0.70}},
        bar="2026-08-12 14:00:00+00:00", signal_audits={}, policy=ExecPolicy(),
        route_fn=lambda *a, **k: ("skip", None, "n/a"), ref_price_fn=lambda _t: None,
        verbose=False)
    assert "pending_fill" not in res.new_managed["UMC"]
    assert "entry_order_id" not in res.new_managed["UMC"]


def test_next_pass_drops_an_entry_that_never_filled():
    """The four Dealer Ranker limits: gone from managed once the broker says 0."""
    managed = {"UMC": {"route": "option", "occ": _UMC, "contracts": 57, "pending_fill": True}}
    res = build_mixed_plan(
        None, targets=[], managed=managed, pos_info={}, bar="2026-08-12 14:00:00+00:00",
        signal_audits={}, policy=ExecPolicy(), route_fn=lambda *a, **k: ("skip", None, "n/a"),
        ref_price_fn=lambda _t: None, verbose=False)
    assert res.new_managed == {}
    assert res.dropped["UMC"]["status"] == "not_found"
