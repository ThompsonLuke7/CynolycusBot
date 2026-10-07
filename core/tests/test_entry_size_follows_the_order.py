"""A settled entry is sized from what its order filled, not from the first look.

`mark_entry_unconfirmed` reads a new order about a second after it is sent. On
2026-09-22 HTF bought 1,264 MYGN at market; the read came back with 51 filled,
the rest filled moments later, and 51 stayed as the position's size. Every sell
is capped at that size, so the exit would have sold 51 and left 1,213 shares
that no module owned. 56 positions were short of the broker this way on
2026-10-05, which is where the account's unowned leftovers came from.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from core.live_4h_exec import ExecPolicy, build_mixed_plan, settle_entry_fill
from core.live_risk_pass import RiskPassConfig, evaluate_risk_exits

NOW = dt.datetime(2026, 9, 22, 14, 30, tzinfo=ZoneInfo("America/New_York"))


class _Client:
    def __init__(self, status, filled_qty, price=3.99, fail=False):
        self._order = {"status": status, "filled_qty": str(filled_qty),
                       "filled_avg_price": str(price)}
        self._fail = fail

    def get_order(self, order_id):
        if self._fail:
            raise RuntimeError("broker unavailable")
        return {"id": order_id, **self._order}


def _mygn(**extra):
    # What mark_entry_unconfirmed left behind: the 51-share first reading.
    return {"route": "equity", "symbol": "MYGN", "shares": 51.0, "remaining_qty": 51.0,
            "entry_filled_qty": 51.0, "entry_fill_price": 3.99, "pending_fill": True,
            "entry_order_id": "order-mygn", **extra}


def test_a_partly_filled_first_reading_is_replaced_by_the_final_fill():
    st = _mygn()

    assert settle_entry_fill(_Client("filled", 1264), st) is True

    assert st["shares"] == st["remaining_qty"] == st["entry_filled_qty"] == 1264.0
    assert "pending_fill" not in st and "entry_order_id" not in st


def test_an_order_still_working_keeps_its_flag_and_is_read_again():
    st = _mygn()

    assert settle_entry_fill(_Client("partially_filled", 400), st) is False

    assert st["shares"] == 400.0
    assert st["pending_fill"] is True and st["entry_order_id"] == "order-mygn"


def test_shares_already_sold_are_not_handed_back():
    st = _mygn(remaining_qty=40.0, shares=40.0)  # 11 of the first 51 already sold

    settle_entry_fill(_Client("filled", 1264), st)

    assert st["entry_filled_qty"] == 1264.0
    assert st["shares"] == st["remaining_qty"] == 1253.0


def test_an_order_that_expired_part_filled_is_sized_at_what_filled():
    st = _mygn()

    assert settle_entry_fill(_Client("expired", 300), st) is True
    assert st["shares"] == 300.0


def test_an_unreadable_order_settles_on_the_broker_position_as_before():
    for client in (None, _Client("filled", 1264, fail=True)):
        st = _mygn()
        assert settle_entry_fill(client, st) is True
        assert st["shares"] == 51.0 and "pending_fill" not in st


def test_the_risk_pass_sizes_the_entry_before_it_caps_a_sell():
    res = evaluate_risk_exits(
        client=_Client("filled", 1264), module="multi_ticker_swing_htf",
        managed={"MYGN": _mygn()},
        pos_info={"MYGN": {"qty": 1264, "avg_entry": 3.99, "current": 4.00}},
        policy=ExecPolicy(target_notional=5000.0), now_et=NOW,
        cfg=RiskPassConfig(hard_stop=False, expiry_flatten=False),
        underlying_fn=lambda _t, at=None: (4.00, 0.2),
    )

    assert res.new_managed["MYGN"]["shares"] == 1264.0
    assert res.confirmed_entries["MYGN"]["qty"] == 1264


def test_the_4h_pass_sizes_the_entry_before_it_plans_an_exit():
    # Past the horizon, so the pass plans a full exit: it must sell all 1,264.
    res = build_mixed_plan(
        _Client("filled", 1264), targets=[], managed={"MYGN": _mygn(runs_held=60, bars_out=60)},
        pos_info={"MYGN": {"qty": 1264, "avg_entry": 3.99, "current": 4.00}},
        bar="2026-09-23 14:00:00+00:00", signal_audits={}, policy=ExecPolicy(),
        route_fn=lambda *a, **k: ("skip", None, "n/a"), ref_price_fn=lambda _t: None,
        verbose=False)

    sells = [p for p in res.plan if p[0] == "MYGN" and p[1] == "sell"]
    assert [p[2] for p in sells] == [1264]
