"""A mark that collapses ~10x in one bar is a data event, not a trading signal.

Regression cover for 2026-08-10. Meta held TENX (100sh @ 15.99) and SION (105sh
@ 48.00). Both marks came back on Monday at roughly a tenth of Friday's close
(13.30 -> 1.42, 49.61 -> 4.46) with the broker's share counts unchanged — the
signature of a forward split whose cost basis was never adjusted. Both tripped
`stop_-39%` and were liquidated for -$1,457 and -$4,572, while no other name in
a 173-position equity book moved worse than -18.9% that session.

The guard cannot tell "unprocessed split" from "the company really did collapse",
and does not try: the correct action is the same for both, which is to hold the
position and escalate rather than sell at a price we cannot vouch for.
"""
from __future__ import annotations

import pytest

from core.live_4h_exec import (
    ExecPolicy,
    IMPLAUSIBLE_EQUITY_BAR_MOVE,
    _implausible_mark_move,
    build_mixed_plan,
)


# --- _implausible_mark_move ----------------------------------------------------

def test_ordinary_down_day_is_credible():
    """08-10's worst genuine equity move was BTDR at -18.9%."""
    assert _implausible_mark_move("equity", 10.88, 8.82) is None


def test_a_hard_but_real_selloff_is_still_credible():
    """-50% in a bar is brutal and real; the guard must not fire on it."""
    assert _implausible_mark_move("equity", 100.0, 50.0) is None


def test_the_tenx_move_is_flagged():
    a = _implausible_mark_move("equity", 13.30, 1.42)
    assert a is not None
    assert a["bar_move"] == pytest.approx(-0.8932, abs=1e-4)
    assert a["ratio"] == pytest.approx(9.37, abs=0.01)
    assert a["direction"] == "down"


def test_the_sion_move_is_flagged():
    a = _implausible_mark_move("equity", 49.61, 4.46)
    assert a is not None
    assert a["ratio"] == pytest.approx(11.12, abs=0.01)


def test_the_ratio_is_reported_raw_not_rounded_to_a_split():
    """No "suspected 9:1 split" field. At these magnitudes consecutive integers
    are only ~10% apart, so any tolerance loose enough to catch a real split
    matches everything — TENX's 9.37 is not a corporate action anyone performed,
    and a confident wrong label is worse than the raw number."""
    a = _implausible_mark_move("equity", 13.30, 1.42)
    assert "split_ratio_suspected" not in a
    assert a["ratio"] == pytest.approx(9.37, abs=0.01)


def test_a_reverse_split_is_flagged_too():
    """1:10 reverse split takes the mark UP 10x with the basis unchanged."""
    a = _implausible_mark_move("equity", 5.0, 50.0)
    assert a is not None
    assert a["bar_move"] == pytest.approx(9.0)
    assert a["ratio"] == pytest.approx(10.0)
    assert a["direction"] == "up"


def test_options_are_exempt():
    """An option premium legitimately loses 84% in a bar — that is the leverage.

    MLTX260821C00020000 did exactly that on 08-10 (1.90 -> 0.30) and its stop
    had to fire. Applying the equity threshold here would block real exits.
    """
    assert _implausible_mark_move("option", 1.90, 0.30) is None


@pytest.mark.parametrize("prev,new", [(None, 5.0), (5.0, None), (0.0, 5.0), (5.0, 0.0),
                                      ("x", 5.0), (5.0, "x")])
def test_unusable_inputs_never_fire(prev, new):
    """No prior mark (first pass) or a junk price means no opinion, not an alert."""
    assert _implausible_mark_move("equity", prev, new) is None


def test_threshold_is_the_documented_constant():
    assert IMPLAUSIBLE_EQUITY_BAR_MOVE == 0.70
    assert _implausible_mark_move("equity", 100.0, 100.0 * (1 - 0.69)) is None
    assert _implausible_mark_move("equity", 100.0, 100.0 * (1 - 0.71)) is not None


# --- build_mixed_plan integration ----------------------------------------------

_POLICY = ExecPolicy()


@pytest.fixture(autouse=True)
def _no_real_bars(tmp_path, monkeypatch):
    """The guard consults the shared 4H cache; keep these tests off the real one."""
    bars = tmp_path / "bars_4h"
    bars.mkdir()
    monkeypatch.setattr("core.live_4h_exec.DEFAULT_BARS_4H_DIR", str(bars))
    return bars


def _write_bars(bars_dir, ticker, closes, start="2026-10-14 18:00:00+00:00"):
    import pandas as pd

    index = pd.date_range(start, periods=len(closes), freq="4h", tz="UTC")
    pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes},
                 index=index).to_parquet(bars_dir / f"{ticker}.parquet")


def _plan(managed, pos_info, targets=()):
    return build_mixed_plan(
        None, targets=list(targets), managed=managed, pos_info=pos_info, bar="2026-08-10 14:00:00+00:00",
        signal_audits={}, policy=_POLICY, route_fn=lambda *a, **k: ("skip", None, "n/a"),
        ref_price_fn=lambda _t: None, verbose=False,
        # Anomaly-guard behavior is independent of the underlying stop; pin the
        # basis off so these assertions never depend on the real bar cache.
        underlying_fn=lambda _t, _at=None: (None, None),
    )


def test_split_does_not_produce_a_stop_order():
    """The whole point: TENX must not be sold into the phantom -91%."""
    managed = {"TENX": {"route": "equity", "symbol": "TENX", "runs_held": 33,
                        "last_mark_price": 13.30, "last_mark_bar": "2026-08-07 18:00:00+00:00"}}
    res = _plan(managed, {"TENX": {"qty": 100, "avg_entry": 15.99, "current": 1.42}})
    assert res.plan == []
    assert "TENX" in res.anomalies
    assert "TENX" in res.new_managed          # still held, still ours
    assert res.dropped == {}


def test_flagged_position_does_not_age_toward_its_horizon():
    """A bar we refused to evaluate must not count as a bar held."""
    managed = {"TENX": {"route": "equity", "symbol": "TENX", "runs_held": 33, "bars_out": 2,
                        "last_mark_price": 13.30}}
    res = _plan(managed, {"TENX": {"qty": 100, "avg_entry": 15.99, "current": 1.42}})
    assert res.new_managed["TENX"]["runs_held"] == 33
    assert res.new_managed["TENX"]["bars_out"] == 2


def test_flagged_position_keeps_its_trusted_prior_mark():
    """last_mark_price must not be overwritten with the suspect value, or the
    next pass would compare bad-to-bad and wave the position straight through."""
    managed = {"TENX": {"route": "equity", "symbol": "TENX", "last_mark_price": 13.30}}
    res = _plan(managed, {"TENX": {"qty": 100, "avg_entry": 15.99, "current": 1.42}})
    assert res.new_managed["TENX"]["last_mark_price"] == 13.30
    assert res.new_managed["TENX"]["mark_anomaly"]["new_mark"] == 1.42


def test_a_normal_stop_still_fires():
    """Guard must not suppress the exits that are supposed to happen."""
    managed = {"AAA": {"route": "equity", "symbol": "AAA", "runs_held": 3,
                       "last_mark_price": 100.0}}
    res = _plan(managed, {"AAA": {"qty": 10, "avg_entry": 100.0, "current": 55.0}})
    assert [(i[0], i[1], i[3]) for i in res.plan] == [("AAA", "sell", "stop_-39%")]
    assert res.anomalies == {}


def test_option_stop_still_fires_through_a_huge_drop():
    """MLTX on 08-10: -84% on the premium, must still be stopped."""
    managed = {"MLTX": {"route": "option", "occ": "MLTX260821C00020000", "runs_held": 10,
                        "last_mark_price": 1.90}}
    res = _plan(managed, {"MLTX260821C00020000": {"qty": 26, "avg_entry": 1.90, "current": 0.30}})
    assert [(i[0], i[3]) for i in res.plan] == [("MLTX260821C00020000", "stop_-39%")]
    assert res.anomalies == {}


def test_anomaly_clears_once_the_mark_is_sane_again():
    """A one-off bad tick must not latch the position out of management."""
    managed = {"AAA": {"route": "equity", "symbol": "AAA", "runs_held": 1,
                       "last_mark_price": 100.0, "mark_anomaly": {"stale": True}}}
    res = _plan(managed, {"AAA": {"qty": 10, "avg_entry": 100.0, "current": 98.0}})
    assert "mark_anomaly" not in res.new_managed["AAA"]
    assert res.anomalies == {}


def test_first_sighting_of_a_position_is_never_flagged():
    """No last_mark_price yet (entry filled this pass) -> nothing to compare."""
    managed = {"AAA": {"route": "equity", "symbol": "AAA", "runs_held": 0}}
    res = _plan(managed, {"AAA": {"qty": 10, "avg_entry": 100.0, "current": 5.0}})
    assert res.anomalies == {}


# --- a move carried by the bars is price action, not a corporate action ---------

_BETR = {"route": "equity", "symbol": "BETR", "runs_held": 20, "entry_avg_price": 9.75,
         "last_mark_price": 8.1945, "last_mark_bar": "2026-10-14 18:00:00+00:00"}


def test_a_rally_across_an_outage_is_evaluated_not_refused(_no_real_bars):
    """BETR 2026-10-26: 8.19 at the last pass before a nine-session outage, 17.68
    after it. The 4H bars climb there (largest bar +27%), so it is price action,
    and the position is past its +30% take-profit."""
    _write_bars(_no_real_bars, "BETR", [8.12, 10.16, 10.50, 12.12, 15.51, 16.61, 17.12, 17.73])

    res = _plan({"BETR": dict(_BETR)}, {"BETR": {"qty": 517, "avg_entry": 9.75, "current": 17.68}})

    assert res.anomalies == {}
    assert [(i[0], i[3]) for i in res.plan] == [("BETR", "take_profit_+30%")]


def test_a_single_bar_break_in_the_bars_is_still_refused(_no_real_bars):
    """The raw cache shows TENX's split as one -90% bar; that must stay a refusal."""
    _write_bars(_no_real_bars, "TENX", [13.25, 14.19, 13.42, 1.37, 1.41, 1.42],
                start="2026-08-06 14:00:00+00:00")
    managed = {"TENX": {"route": "equity", "symbol": "TENX", "runs_held": 33,
                        "last_mark_price": 13.30, "last_mark_bar": "2026-08-07 14:00:00+00:00"}}

    res = _plan(managed, {"TENX": {"qty": 100, "avg_entry": 15.99, "current": 1.42}})

    assert res.plan == [] and "TENX" in res.anomalies


@pytest.mark.parametrize("closes", [
    None,                                    # no bars at all
    [8.12, 17.73],                           # too few to show a path
    [8.12, 9.0, 10.0, 11.0],                 # bars end far from the mark
    [4.00, 8.0, 12.0, 16.0, 17.7],           # bars do not start at the trusted mark
])
def test_bars_that_do_not_cover_the_move_leave_the_refusal_in_place(_no_real_bars, closes):
    if closes is not None:
        _write_bars(_no_real_bars, "BETR", closes)

    res = _plan({"BETR": dict(_BETR)}, {"BETR": {"qty": 517, "avg_entry": 9.75, "current": 17.68}})

    assert res.plan == [] and "BETR" in res.anomalies
