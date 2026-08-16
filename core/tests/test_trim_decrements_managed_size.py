"""A partial exit must reduce the size persisted in managed state.

`build_mixed_plan` re-reads the live quantity from the broker every run, so a
stale size never mis-sizes an order and this was invisible to the trading path.
It is not invisible to anything that values the open book from state: on
2026-08-13 the Dealer Ranker trimmed 30 of 61 FIG contracts, kept saying 61, and
overstated that one position by $6,360 in the daily report.
"""
from __future__ import annotations

from core.live_4h_exec import ExecPolicy, build_mixed_plan


def _no_entries(*args, **kwargs):
    raise AssertionError("this test manages an existing position; no entry routing expected")


def _plan(managed, pos_info, targets=("AAA",)):
    return build_mixed_plan(
        object(),
        targets=list(targets),
        managed=managed,
        pos_info=pos_info,
        bar="2026-08-13 18:00:00+00:00",
        signal_audits={},
        policy=ExecPolicy(),
        route_fn=_no_entries,
        ref_price_fn=lambda ticker: 10.0,
        verbose=False,
    )


def test_option_trim_decrements_contracts():
    # take_profit 0.30 with scale_frac 0.16 -> trim floor(0.16 * 61) = 9 of 61.
    managed = {"AAA": {"route": "option", "occ": "AAA260821C00023500", "contracts": 61,
                       "runs_held": 1, "bars_out": 0, "trimmed": False,
                       "entry_bar": "2026-08-12 18:00:00+00:00"}}
    res = _plan(managed, {"AAA260821C00023500": {"qty": 61, "avg_entry": 0.80, "current": 2.92}})

    sells = [p for p in res.plan if p[1] == "sell"]
    assert len(sells) == 1
    sold_qty = sells[0][2]
    assert sold_qty == 9
    state = res.new_managed["AAA"]
    assert state["trimmed"] is True
    assert state["contracts"] == 61 - sold_qty


def test_equity_trim_decrements_shares():
    managed = {"AAA": {"route": "equity", "symbol": "AAA", "shares": 100,
                       "runs_held": 1, "bars_out": 0, "trimmed": False,
                       "entry_bar": "2026-08-12 18:00:00+00:00"}}
    res = _plan(managed, {"AAA": {"qty": 100, "avg_entry": 10.0, "current": 14.0}})

    sold_qty = [p for p in res.plan if p[1] == "sell"][0][2]
    assert sold_qty == 16
    assert res.new_managed["AAA"]["shares"] == 100 - sold_qty


def test_trim_uses_broker_quantity_not_state_quantity():
    """State already drifted below the broker: the trim sizes off the broker.

    This is the invariant that kept the bug out of the trading path, so it must
    survive the fix — the decrement is applied to what was actually sold.
    """
    managed = {"AAA": {"route": "option", "occ": "AAA260821C00023500", "contracts": 10,
                       "runs_held": 1, "bars_out": 0, "trimmed": False,
                       "entry_bar": "2026-08-12 18:00:00+00:00"}}
    res = _plan(managed, {"AAA260821C00023500": {"qty": 61, "avg_entry": 0.80, "current": 2.92}})

    sold_qty = [p for p in res.plan if p[1] == "sell"][0][2]
    assert sold_qty == 9              # floor(0.16 * 61), from the broker
    assert res.new_managed["AAA"]["contracts"] == 1   # 10 - 9, never negative


def test_trim_size_never_goes_negative():
    managed = {"AAA": {"route": "option", "occ": "AAA260821C00023500", "contracts": 2,
                       "runs_held": 1, "bars_out": 0, "trimmed": False,
                       "entry_bar": "2026-08-12 18:00:00+00:00"}}
    res = _plan(managed, {"AAA260821C00023500": {"qty": 61, "avg_entry": 0.80, "current": 2.92}})
    assert res.new_managed["AAA"]["contracts"] == 0


def test_full_exit_leaves_no_managed_state_to_decrement():
    """A stop is a full exit — the position leaves `new_managed` entirely."""
    managed = {"AAA": {"route": "option", "occ": "AAA260821C00023500", "contracts": 61,
                       "runs_held": 1, "bars_out": 0, "trimmed": False,
                       "entry_bar": "2026-08-12 18:00:00+00:00"}}
    # targets=() so the fully-exited name is not immediately routed as a new entry.
    res = _plan(managed, {"AAA260821C00023500": {"qty": 61, "avg_entry": 0.80, "current": 0.40}},
                targets=())

    assert [p[3] for p in res.plan] == ["stop_-39%"]
    assert "AAA" not in res.new_managed
    assert "AAA260821C00023500" in res.exit_context
