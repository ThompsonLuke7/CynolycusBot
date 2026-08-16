"""An ACCEPTED exit order is not a closed position. Confirm the fill first.

2026-08-14: Dealer Ranker stopped out HPE260814C00060000 (52 ctr) and
ZM260814C00110000 (26 ctr) at 15:53 ET. Both were 0DTE with no bid, so the
market sell was rejected ("no available quote") and the ladder repriced to its
$0.01 floor. The broker ACCEPTED both orders; neither could ever fill.

execute_plan treated acceptance as completion: it dropped both from managed and
wrote `realized-PnL logged: ... pnl=None` four seconds later. Consequences:

  * The -$4,784 / -$4,810 losses never reached the realized ledger. 43 of 62
    rows in dealer_ranker/closed_trades.jsonl were written this way (69%).
  * The positions became unowned broker rows, so the swing module adopted them
    at 15:53 and fired its own exits — 403 "not eligible to trade uncovered
    option contracts", then 422 "contract is expired".

The fix keeps the position owned and unbooked until the broker settles it. The
$0.01 rung stays as-is: there is nowhere lower to go, and an OTM contract with
an empty book is not coming back.
"""
from __future__ import annotations

import json

from core.live_4h_exec import (
    ExecPolicy,
    build_mixed_plan,
    execute_plan,
    resolve_settled_exit,
)

_HPE = "HPE260814C00060000"


class _Client:
    """Accepts the sell but never fills it — the 0DTE-with-no-bid case."""

    def __init__(self, *, fill_price=None, order_status="accepted"):
        self.fill_price = fill_price
        self.order_status = order_status
        self.submitted: list[dict] = []

    def submit_option_order(self, *, symbol, qty, side, **k):
        self.submitted.append({"symbol": symbol, "qty": qty, "side": side, **k})
        return {"id": f"{symbol}-oid", "status": "accepted", "filled_avg_price": None}

    def get_order(self, order_id):
        return {"id": order_id, "status": self.order_status,
                "filled_avg_price": self.fill_price}

    def get_positions(self):
        return []


def _managed():
    return {"HPE": {"route": "option", "occ": _HPE, "contracts": 52,
                    "entry_bar": "2026-08-12T19:52:00+00:00", "runs_held": 2,
                    "entry_avg_price": 0.92}}


def _exec(client, new_managed, exit_context, ledger_root, monkeypatch):
    # execute_plan defers exits when the market is closed, which depends on wall
    # clock. Neutralise it so these tests exercise the submit path deterministically.
    monkeypatch.setattr("core.live_4h_exec.defer_exits_if_opg_unavailable",
                        lambda module, bar, plan, limits, **kw: plan)
    return execute_plan(
        client,
        plan=[(_HPE, "sell", 52, "stop_-50%", "option")],
        limits={}, submit=True, equity_tif_fn=lambda: "day",
        new_managed=new_managed, exit_context=exit_context,
        module="dealer_ranker", pos_lookup={_HPE: {"avg_entry": 0.92}},
        bar="2026-08-14 19:45", ledger_root=str(ledger_root),
    )


def _ledger_rows(ledger_root):
    path = ledger_root / "dealer_ranker" / "closed_trades.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_accepted_but_unfilled_exit_is_not_booked_as_a_close(tmp_path, monkeypatch):
    client = _Client(fill_price=None)
    new_managed: dict = {}
    exit_context = {_HPE: ("HPE", _managed()["HPE"])}

    _exec(client, new_managed, exit_context, tmp_path, monkeypatch)

    # No phantom row: the ledger stays empty rather than gaining a pnl=None close.
    assert _ledger_rows(tmp_path) == []
    # The position stays OURS so no sibling module can adopt it.
    assert "HPE" in new_managed
    pending = new_managed["HPE"]["exit_pending"]
    assert pending["order_id"] == f"{_HPE}-oid"
    assert pending["reason"] == "stop_-50%"
    # Basis is snapshotted now, because once it settles the broker cannot supply it.
    assert pending["entry_avg_price"] == 0.92


def test_filled_exit_still_books_immediately(tmp_path, monkeypatch):
    client = _Client(fill_price="0.46")
    new_managed: dict = {}
    exit_context = {_HPE: ("HPE", _managed()["HPE"])}

    _exec(client, new_managed, exit_context, tmp_path, monkeypatch)

    rows = _ledger_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["realized_pnl"] == round((0.46 - 0.92) * 100 * 52, 2)
    assert "HPE" not in new_managed


def test_settled_pending_exit_books_the_expiry_loss(tmp_path):
    """Gone from the broker with no fill == the premium expired worthless."""
    client = _Client(fill_price=None, order_status="expired")
    state = {**_managed()["HPE"], "exit_pending": {
        "order_id": f"{_HPE}-oid", "reason": "stop_-50%", "route": "option",
        "qty": 52.0, "entry_avg_price": 0.92, "submitted_bar": "2026-08-14 19:45"}}

    booked = resolve_settled_exit(client, module="dealer_ranker", ticker="HPE",
                                  symbol=_HPE, state=state, bar="2026-08-15 13:45",
                                  ledger_root=str(tmp_path))

    assert booked is True
    rows = _ledger_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["settle_outcome"] == "expired_worthless"
    # The full premium is the loss: 52 contracts x 100 x $0.92.
    assert rows[0]["realized_pnl"] == -4784.0


def test_settled_pending_exit_books_a_late_fill_at_its_real_price(tmp_path):
    client = _Client(fill_price="0.05", order_status="filled")
    state = {**_managed()["HPE"], "exit_pending": {
        "order_id": f"{_HPE}-oid", "reason": "stop_-50%", "route": "option",
        "qty": 52.0, "entry_avg_price": 0.92, "submitted_bar": "2026-08-14 19:45"}}

    assert resolve_settled_exit(client, module="dealer_ranker", ticker="HPE",
                                symbol=_HPE, state=state, bar="2026-08-15 13:45",
                                ledger_root=str(tmp_path)) is True
    rows = _ledger_rows(tmp_path)
    assert rows[0]["settle_outcome"] == "exit_filled"
    assert rows[0]["realized_pnl"] == round((0.05 - 0.92) * 100 * 52, 2)


def test_build_mixed_plan_settles_a_pending_exit_when_the_broker_goes_flat(tmp_path):
    client = _Client(fill_price=None, order_status="expired")
    managed = {"HPE": {**_managed()["HPE"], "exit_pending": {
        "order_id": f"{_HPE}-oid", "reason": "stop_-50%", "route": "option",
        "qty": 52.0, "entry_avg_price": 0.92, "submitted_bar": "2026-08-14 19:45"}}}

    res = build_mixed_plan(
        client, targets=[], managed=managed, pos_info={}, bar="2026-08-15 13:45",
        signal_audits={}, policy=ExecPolicy(), route_fn=lambda *a, **k: None,
        ref_price_fn=lambda t: None, verbose=False,
        module="dealer_ranker", ledger_root=str(tmp_path),
    )

    assert res.dropped["HPE"]["exit_settled"] is True
    assert _ledger_rows(tmp_path)[0]["realized_pnl"] == -4784.0


def test_still_held_after_a_failed_exit_is_surfaced_as_stuck(tmp_path):
    """The order did not fill and the contract is still there: re-plan, and shout."""
    client = _Client(fill_price=None)
    managed = {"HPE": {**_managed()["HPE"], "exit_pending": {
        "order_id": f"{_HPE}-oid", "reason": "stop_-50%", "route": "option",
        "qty": 52.0, "entry_avg_price": 0.92, "submitted_bar": "2026-08-14 19:45"}}}

    res = build_mixed_plan(
        client, targets=[], managed=managed, pos_info={_HPE: {"qty": 52, "current": 0.01}},
        bar="2026-08-14 19:50", signal_audits={}, policy=ExecPolicy(),
        route_fn=lambda *a, **k: None, ref_price_fn=lambda t: None, verbose=False,
        module="dealer_ranker", ledger_root=str(tmp_path),
    )

    assert "HPE" in res.stuck_exits
    assert res.stuck_exits["HPE"]["reason"] == "stop_-50%"
    # Flag cleared so the exit machine re-evaluates rather than sitting on a stale order.
    assert "exit_pending" not in res.new_managed.get("HPE", {})
    # Nothing booked: the position is still held.
    assert _ledger_rows(tmp_path) == []
