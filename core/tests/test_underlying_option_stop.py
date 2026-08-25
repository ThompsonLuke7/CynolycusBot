"""Option stops are measured on the UNDERLYING, not the contract premium.

The premium stop (`stop_loss=0.39`) was selected on a shares-only backtest and
then applied to option premium, where ~-13% of underlying is a -39% premium move.
Across 42 live option stops with resolvable 4H bars (2026-07-17..08-18) the median
underlying move at the moment the premium stop fired was -3.1%, and 18 of 42 fired
while the underlying was down less than 2% — LITE on 2026-08-12 stopped for -$4,950
with its underlying at +0.04%. See research/daily_live_reports/underlying_vs_premium_stop.md.

These lock in the replacement AND its fail-safes: equity is untouched, and an
option whose underlying basis cannot be established keeps the premium stop.
"""
from __future__ import annotations

import pytest

from core.live_4h_exec import (
    ExecPolicy,
    build_mixed_plan,
    exit_action,
    underlying_basis,
    underlying_stop_level,
)

P = ExecPolicy()  # live defaults: stop_loss=0.39, underlying_stop_atr=1.5


# --- exit_action ---------------------------------------------------------------

def test_premium_collapse_is_held_while_the_underlying_holds():
    """The LITE case: premium -97%, underlying flat. Not a thesis failure."""
    act = exit_action(-0.97, runs_held=2, bars_out=0, trimmed=False, policy=P,
                      route="option", u_entry=100.0, u_now=100.04, u_atr=4.67)
    assert act == ("hold", "")


def test_underlying_break_exits_even_while_the_premium_looks_fine():
    """Thesis is broken on the underlying; the contract's mark is irrelevant."""
    act = exit_action(-0.05, runs_held=2, bars_out=0, trimmed=False, policy=P,
                      route="option", u_entry=100.0, u_now=92.0, u_atr=5.0)
    assert act == ("exit", "underlying_stop_-1.5atr")


def test_the_stop_is_exactly_n_atrs_below_entry():
    for u_now, expect_exit in [(92.51, False), (92.50, True), (92.49, True)]:
        act = exit_action(-0.20, runs_held=2, bars_out=0, trimmed=False, policy=P,
                          route="option", u_entry=100.0, u_now=u_now, u_atr=5.0)
        assert (act[0] == "exit") is expect_exit, u_now


def test_equity_is_untouched_and_still_uses_the_premium_stop():
    """Equity is the only currently-profitable route; do not change it."""
    act = exit_action(-0.45, runs_held=2, bars_out=0, trimmed=False, policy=P,
                      route="equity", u_entry=100.0, u_now=99.0, u_atr=5.0)
    assert act == ("exit", "stop_-39%")


@pytest.mark.parametrize("u_entry,u_now,u_atr", [
    (None, 100.0, 5.0),      # never anchored
    (100.0, None, 5.0),      # no current price this pass
    (100.0, 100.0, None),    # no ATR
    (100.0, 100.0, 0.0),     # degenerate ATR
    (0.0, 100.0, 5.0),       # junk entry price
])
def test_missing_basis_falls_back_to_the_premium_stop(u_entry, u_now, u_atr):
    """A half-known basis must never silently widen the stop to infinity."""
    act = exit_action(-0.45, runs_held=2, bars_out=0, trimmed=False, policy=P,
                      route="option", u_entry=u_entry, u_now=u_now, u_atr=u_atr)
    assert act == ("exit", "stop_-39%")


def test_callers_that_pass_no_basis_get_the_old_behavior_exactly():
    """Back-compat: the pre-2026-08-18 call signature is unchanged in effect."""
    assert exit_action(-0.45, runs_held=2, bars_out=0, trimmed=False,
                       policy=P) == ("exit", "stop_-39%")


def test_the_dte_rail_is_OFF_by_default():
    """Live setting. core.live_risk_pass already flattens on the last tradable
    session, and Dealer Ranker deliberately buys near-dated weeklies — a DTE
    floor here would enter and exit those on consecutive runs. Near-expiry
    contracts are also where this book's largest gains have come from (BLZE
    +168%, SNDK +454%, MRVL +131% peaks, all 08-21 weeklies)."""
    assert P.min_dte_exit is None
    assert exit_action(-0.60, runs_held=2, bars_out=0, trimmed=False, policy=P,
                       route="option", u_entry=100.0, u_now=99.0, u_atr=5.0,
                       dte=0)[0] == "hold"


def test_the_dte_rail_works_when_a_module_opts_in():
    p = ExecPolicy(min_dte_exit=5)
    assert exit_action(-0.60, runs_held=2, bars_out=0, trimmed=False, policy=p,
                       route="option", u_entry=100.0, u_now=99.0, u_atr=5.0,
                       dte=5) == ("exit", "expiry_dte<=5")
    assert exit_action(-0.10, runs_held=2, bars_out=0, trimmed=False, policy=p,
                       route="option", u_entry=100.0, u_now=99.0, u_atr=5.0,
                       dte=6)[0] == "hold"
    assert exit_action(-0.10, runs_held=2, bars_out=0, trimmed=False, policy=p,
                       route="equity", dte=0)[0] == "hold"


def test_underlying_break_outranks_the_expiry_guard():
    act = exit_action(-0.10, runs_held=2, bars_out=0, trimmed=False,
                      policy=ExecPolicy(min_dte_exit=5),
                      route="option", u_entry=100.0, u_now=90.0, u_atr=5.0, dte=1)
    assert act == ("exit", "underlying_stop_-1.5atr")


def test_take_profit_still_trims_on_premium_gain():
    """Trims are 32-for-32 winners live; the change must not touch them."""
    act = exit_action(0.35, runs_held=2, bars_out=0, trimmed=False, policy=P,
                      route="option", u_entry=100.0, u_now=101.0, u_atr=5.0, dte=30)
    assert act == ("trim", "take_profit_+30%")


def test_setting_underlying_stop_atr_to_none_restores_premium_stops():
    p = ExecPolicy(underlying_stop_atr=None)
    act = exit_action(-0.45, runs_held=2, bars_out=0, trimmed=False, policy=p,
                      route="option", u_entry=100.0, u_now=100.0, u_atr=5.0)
    assert act == ("exit", "stop_-39%")


# --- underlying_stop_level / underlying_basis ----------------------------------

def test_stop_level_arithmetic():
    assert underlying_stop_level(P, 100.0, 4.0) == pytest.approx(94.0)


@pytest.mark.parametrize("u_entry,u_atr", [(None, 4.0), (100.0, None), (0.0, 4.0), (100.0, 0.0)])
def test_stop_level_is_none_without_a_usable_basis(u_entry, u_atr):
    assert underlying_stop_level(P, u_entry, u_atr) is None


def test_basis_is_none_for_an_unknown_ticker():
    assert underlying_basis("ZZZZ_NOT_A_TICKER") == (None, None)


def test_basis_is_none_when_history_is_shorter_than_the_atr_window(tmp_path):
    import pandas as pd
    idx = pd.date_range("2026-08-01", periods=5, freq="4h", tz="UTC")
    pd.DataFrame({"high": 1.0, "low": 0.9, "close": 0.95}, index=idx).to_parquet(
        tmp_path / "SHORT.parquet")
    assert underlying_basis("SHORT", bars_dir=tmp_path) == (None, None)


def test_basis_at_a_timestamp_never_reads_a_later_bar(tmp_path):
    import pandas as pd
    idx = pd.date_range("2026-08-01", periods=40, freq="4h", tz="UTC")
    closes = list(range(100, 140))
    pd.DataFrame({"high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
                  "close": closes}, index=idx).to_parquet(tmp_path / "RAMP.parquet")
    px, atr = underlying_basis("RAMP", idx[20], bars_dir=tmp_path)
    assert px == pytest.approx(120.0)          # the bar AT that stamp, not the last one
    assert atr == pytest.approx(2.0)
    assert underlying_basis("RAMP", idx[0] - pd.Timedelta("1D"),
                            bars_dir=tmp_path) == (None, None)   # predates the file


# --- build_mixed_plan integration ----------------------------------------------

def _plan(managed, pos_info, *, ufn, policy=P):
    return build_mixed_plan(
        None, targets=[], managed=managed, pos_info=pos_info,
        bar="2026-08-18 14:00:00+00:00", signal_audits={}, policy=policy,
        route_fn=lambda *a, **k: ("skip", None, "n/a"),
        ref_price_fn=lambda _t: None, verbose=False, underlying_fn=ufn,
    )


def _held(**over):
    st = {"route": "option", "occ": "XYZ260918C00100000", "contracts": 5, "runs_held": 3,
          "bars_out": 0, "trimmed": False, "entry_bar": "2026-08-12 18:00:00+00:00",
          "expiry": "2026-09-18", "last_mark_price": 9.20}
    st.update(over)
    return {"XYZ": st}


def test_plan_backfills_the_basis_from_the_entry_bar_and_holds():
    calls = []
    def ufn(t, at=None):
        calls.append(at)
        return (100.0, 5.0) if at else (99.0, 5.0)   # entry 100, now 99 -> above 92.5
    res = _plan(_held(), {"XYZ260918C00100000": {"qty": 5, "avg_entry": 9.20, "current": 4.00}},
                ufn=ufn)
    assert res.plan == []                                   # -57% premium, held
    st = res.new_managed["XYZ"]
    assert st["u_entry"] == 100.0 and st["u_atr"] == 5.0
    assert st["u_basis_source"] == "backfilled_from_entry_bar"
    assert calls[0] == "2026-08-12 18:00:00+00:00"          # anchored at ENTRY, not now


def test_plan_stops_out_when_the_underlying_breaks():
    res = _plan(_held(u_entry=100.0, u_atr=5.0),
                {"XYZ260918C00100000": {"qty": 5, "avg_entry": 9.20, "current": 8.90}},
                ufn=lambda _t, at=None: (100.0, 5.0) if at else (91.0, 5.0))
    assert [(i[0], i[3]) for i in res.plan] == [("XYZ260918C00100000",
                                                 "underlying_stop_-1.5atr")]


def test_a_stored_basis_is_never_re_derived_from_todays_price():
    """Re-reading the anchor each pass would let the stop chase price down."""
    res = _plan(_held(u_entry=100.0, u_atr=5.0),
                {"XYZ260918C00100000": {"qty": 5, "avg_entry": 9.20, "current": 4.00}},
                ufn=lambda _t, at=None: (60.0, 3.0) if at else (91.0, 5.0))
    assert [(i[0], i[3]) for i in res.plan] == [("XYZ260918C00100000",
                                                 "underlying_stop_-1.5atr")]
    assert res.new_managed == {}          # exited, and never re-anchored at 60


def test_a_position_with_no_entry_bar_keeps_the_premium_stop():
    """Backfilling with `at=None` would anchor at TODAY, creating a stop that can
    never fire. No entry_bar must mean no basis."""
    st = _held(); st["XYZ"].pop("entry_bar")
    res = _plan(st, {"XYZ260918C00100000": {"qty": 5, "avg_entry": 9.20, "current": 4.00}},
                ufn=lambda _t, at=None: (100.0, 5.0))
    assert [(i[0], i[3]) for i in res.plan] == [("XYZ260918C00100000", "stop_-39%")]
    assert "u_entry" not in res.new_managed.get("XYZ", {})


def test_an_unresolvable_underlying_keeps_the_premium_stop():
    res = _plan(_held(), {"XYZ260918C00100000": {"qty": 5, "avg_entry": 9.20, "current": 4.00}},
                ufn=lambda _t, at=None: (None, None))
    assert [(i[0], i[3]) for i in res.plan] == [("XYZ260918C00100000", "stop_-39%")]


def test_expiry_rail_fires_from_the_stored_expiry_when_a_module_opts_in():
    res = _plan(_held(expiry="2026-08-21", u_entry=100.0, u_atr=5.0),
                {"XYZ260918C00100000": {"qty": 5, "avg_entry": 9.20, "current": 4.00}},
                ufn=lambda _t, at=None: (99.0, 5.0), policy=ExecPolicy(min_dte_exit=5))
    assert [(i[0], i[3]) for i in res.plan] == [("XYZ260918C00100000", "expiry_dte<=5")]


def test_a_near_expiry_winner_is_left_alone_under_the_live_policy():
    """The BLZE shape: 3 DTE, deep in profit, underlying far above the stop.
    It trims and rides rather than being closed by a calendar rail."""
    res = _plan(_held(expiry="2026-08-21", u_entry=100.0, u_atr=5.0),
                {"XYZ260918C00100000": {"qty": 25, "avg_entry": 9.20, "current": 24.0}},
                ufn=lambda _t, at=None: (118.0, 5.0))
    assert [(i[0], i[3]) for i in res.plan] == [("XYZ260918C00100000",
                                                 "take_profit_+30%")]


def test_a_small_option_position_books_a_trim_instead_of_nothing():
    """Was the defect: `floor(0.16 x 4)` = 0, so a 4-contract winner sold nothing.
    See trim_quantity for the MRVL/SNDK cases that motivated the fix."""
    res = _plan(_held(u_entry=100.0, u_atr=5.0),
                {"XYZ260918C00100000": {"qty": 4, "avg_entry": 9.20, "current": 24.0}},
                ufn=lambda _t, at=None: (118.0, 5.0))
    assert [(i[2], i[3]) for i in res.plan] == [(1, "take_profit_+30%")]
    assert res.new_managed["XYZ"]["trimmed"] is True


# --- the between-bar risk pass must mirror the same stop ------------------------
# This pass runs every few minutes against the 4H runner's twice a day, so it is
# the path that actually fires most stops. If it kept a premium stop the 4H
# change would be cosmetic.

def _risk(st, cur, *, ufn=lambda _t, _at=None: (91.0, 5.0)):
    import datetime as dt
    from core.live_risk_pass import evaluate_risk_exits
    return evaluate_risk_exits(
        None, module="t", managed={"XYZ": st}, policy=P,
        pos_info={"XYZ260918C00100000": {"qty": 5, "avg_entry": 9.20, "current": cur}},
        now_et=dt.datetime(2026, 8, 18, 11, 0), underlying_fn=ufn)


def _anchored(**over):
    st = {"route": "option", "occ": "XYZ260918C00100000", "contracts": 5,
          "entry_bar": "2026-08-12 18:00:00+00:00", "expiry": "2026-09-18",
          "last_mark_price": 9.20, "u_entry": 100.0, "u_atr": 5.0}
    st.update(over)
    return st


def test_risk_pass_holds_a_premium_collapse_while_the_underlying_holds():
    res = _risk(_anchored(), 4.00, ufn=lambda _t, _at=None: (99.0, 5.0))
    assert res.plan == []


def test_risk_pass_stops_out_on_a_real_underlying_break():
    res = _risk(_anchored(), 8.90)          # premium fine, underlying broken
    assert [(i[0], i[3]) for i in res.plan] == [("XYZ260918C00100000",
                                                 "underlying_stop_-1.5atr")]


def test_risk_pass_keeps_the_premium_stop_when_the_runner_never_anchored():
    st = _anchored(); st.pop("u_entry"); st.pop("u_atr")
    res = _risk(st, 4.00)
    assert [(i[0], i[3]) for i in res.plan] == [("XYZ260918C00100000", "stop_-39%")]


def test_risk_pass_never_invents_a_basis():
    """Only the 4H runner sees entries, so only it may anchor a position."""
    st = _anchored(); st.pop("u_entry"); st.pop("u_atr")
    called = []
    _risk(st, 4.00, ufn=lambda t, at=None: called.append(t) or (100.0, 5.0))
    assert called == []


def test_risk_pass_leaves_equity_on_the_premium_stop():
    import datetime as dt
    from core.live_risk_pass import evaluate_risk_exits
    res = evaluate_risk_exits(
        None, module="t",
        managed={"AAA": {"route": "equity", "symbol": "AAA", "last_mark_price": 100.0}},
        pos_info={"AAA": {"qty": 10, "avg_entry": 100.0, "current": 55.0}},
        policy=P, now_et=dt.datetime(2026, 8, 18, 11, 0))
    assert [(i[0], i[3]) for i in res.plan] == [("AAA", "stop_-39%")]


# --- trim sizing: never silently zero, never a phantom open position -----------
# `int(scale_frac * held)` floored is 0 for any option smaller than 1/scale_frac
# contracts. Live 2026-08-18: MRVL held 4 against scale_frac 0.16, peaked +131%,
# fell to -15%, booked nothing. SNDK held 1 and peaked +454%, likewise.

from core.live_4h_exec import is_partial_trim, take_profit_reason, trim_quantity


@pytest.mark.parametrize("held,expect", [(2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (22, 3), (100, 16)])
def test_a_trim_always_sells_at_least_one_unit(held, expect):
    assert trim_quantity(0.16, held) == expect


def test_a_single_unit_position_returns_the_whole_position():
    """Signals a full take-profit — 1 contract cannot be split."""
    assert trim_quantity(0.16, 1) == 1
    assert trim_quantity(0.50, 1) == 1


def test_scale_frac_one_returns_the_whole_position():
    assert trim_quantity(1.0, 22) == 22


def test_trim_never_exceeds_what_is_held():
    assert trim_quantity(5.0, 3) == 3


@pytest.mark.parametrize("held", [0, -4, None, "x"])
def test_unusable_size_trims_nothing(held):
    assert trim_quantity(0.16, held) == 0


def test_full_and_partial_take_profit_have_distinct_reasons():
    """They must not share a string: every downstream reader infers 'still open'
    from a take_profit_ row, so a full exit wearing the trim's reason would be
    counted as an open runner forever."""
    p = ExecPolicy(take_profit=0.30)
    assert take_profit_reason(p, full=False) == "take_profit_+30%"
    assert take_profit_reason(p, full=True) == "take_profit_full_+30%"
    assert is_partial_trim("take_profit_+30%") is True
    assert is_partial_trim("take_profit_full_+30%") is False
    assert is_partial_trim("stop_-39%") is False
    assert is_partial_trim(None) is False


def test_the_mrvl_case_now_books_a_trim(_ufn=lambda _t, at=None: (100.0, 5.0)):
    """4 contracts at +131%: was 0, now sells 1 and keeps riding 3."""
    res = _plan(_held(u_entry=100.0, u_atr=5.0),
                {"XYZ260918C00100000": {"qty": 4, "avg_entry": 10.0, "current": 23.1}},
                ufn=_ufn)
    assert [(i[0], i[2], i[3]) for i in res.plan] == [
        ("XYZ260918C00100000", 1, "take_profit_+30%")]
    st = res.new_managed["XYZ"]
    assert st["trimmed"] is True and st["contracts"] == 4   # 5 held - 1 sold


def test_the_sndk_case_takes_profit_in_full():
    """1 contract cannot be trimmed; take the profit rather than book nothing."""
    res = _plan(_held(contracts=1, u_entry=100.0, u_atr=5.0),
                {"XYZ260918C00100000": {"qty": 1, "avg_entry": 10.0, "current": 55.4}},
                ufn=lambda _t, at=None: (100.0, 5.0))
    assert [(i[0], i[2], i[3]) for i in res.plan] == [
        ("XYZ260918C00100000", 1, "take_profit_full_+30%")]
    assert "XYZ" not in res.new_managed          # closed, not a phantom runner
    assert "XYZ260918C00100000" in res.exit_context


def test_a_full_take_profit_is_never_marked_trimmed():
    """Marking it trimmed AND selling everything would strand an open position."""
    res = _plan(_held(contracts=1, u_entry=100.0, u_atr=5.0),
                {"XYZ260918C00100000": {"qty": 1, "avg_entry": 10.0, "current": 55.4}},
                ufn=lambda _t, at=None: (100.0, 5.0))
    assert res.new_managed == {}


def test_equity_trims_are_unaffected_by_the_rounding_fix():
    """0.16 x 100 shares = 16 either way; the shares path must not move."""
    res = _plan({"AAA": {"route": "equity", "symbol": "AAA", "shares": 100,
                         "runs_held": 3, "bars_out": 0, "trimmed": False,
                         "entry_bar": "2026-08-12 18:00:00+00:00",
                         "last_mark_price": 10.0}},
                {"AAA": {"qty": 100, "avg_entry": 10.0, "current": 13.5}},
                ufn=lambda _t, at=None: (None, None))
    assert [(i[0], i[2], i[3]) for i in res.plan] == [("AAA", 16, "take_profit_+30%")]
