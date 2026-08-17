"""The between-bar risk pass: enforce the stop, change nothing else.

2026-08-14 motivated this. Dealer Ranker runs once a day at 15:45 ET, so HPE and
ZM — bought 8/12 and 8/13, both expiring 8/14 — each got exactly ONE stop test in
their entire life, minutes before expiry with no bid left. They expired worthless
for -$9,594.

The danger in fixing that is doing too much. A pass that runs 78 times a day
instead of twice must not:

  * age positions toward the horizon exit (runs_held/bars_out count 4H BARS),
  * evaluate rank drop-out (a model output),
  * silently re-tune the trailing stop by sampling more peaks than the 4H
    calibration ever saw.

These tests pin all three.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from core.live_4h_exec import ExecPolicy
from core.live_risk_pass import (
    RiskPassConfig,
    evaluate_risk_exits,
    expiring_before_next_session,
    module_state_lock,
    parse_occ_expiry,
    risk_exit_action,
)

ET = ZoneInfo("America/New_York")
HPE = "HPE260814C00060000"
DEALER = ExecPolicy(take_profit=0.20, scale_frac=0.5, horizon_bars=25,
                    stop_loss=0.50, trail_stop=0.35)


class _Client:
    """No option quote available, so mark corroboration is a no-op."""

    def __init__(self, quote=None):
        self.quote = quote
        self.submitted: list[dict] = []

    def get_option_quotes(self, symbols, limit=1):
        return {"quotes": {symbols: self.quote}} if self.quote else {}

    def get_positions(self):
        return []


def _managed(**over):
    base = {"route": "option", "occ": HPE, "contracts": 52,
            "entry_bar": "2026-08-12T19:52:00+00:00", "runs_held": 2,
            "bars_out": 1, "entry_avg_price": 0.92, "last_mark_price": 0.92}
    base.update(over)
    return {"HPE": base}


def _pos(qty=52, avg=0.92, cur=0.40):
    return {HPE: {"qty": qty, "avg_entry": avg, "current": cur}}


def _eval(managed, pos_info, *, now=None, cfg=None, policy=DEALER, client=None):
    return evaluate_risk_exits(
        client or _Client(), module="dealer_ranker", managed=managed,
        pos_info=pos_info, policy=policy,
        now_et=now or dt.datetime(2026, 8, 13, 11, 0, tzinfo=ET),
        cfg=cfg or RiskPassConfig())


# --- the whole point: a stop is enforced between bars -------------------------

def test_hard_stop_fires_between_bars():
    res = _eval(_managed(), _pos(cur=0.40))  # -56.5% against a 50% stop
    assert res.plan == [(HPE, "sell", 52, "stop_-50%", "option")]
    assert HPE in res.exit_context


def test_position_above_the_stop_is_left_alone():
    res = _eval(_managed(), _pos(cur=0.80))  # -13%
    assert res.plan == []


# --- the danger: this pass must not age or re-rank anything -------------------

def test_bar_counters_never_advance():
    """runs_held/bars_out count 4H bars. 78 ticks a day must not touch them."""
    managed = _managed()
    for _ in range(20):
        res = _eval(managed, _pos(cur=0.80))
        managed = res.new_managed
    assert managed["HPE"]["runs_held"] == 2
    assert managed["HPE"]["bars_out"] == 1


def test_horizon_is_never_evaluated_here():
    """A position past its horizon is the 4H runner's call, not this pass's."""
    res = _eval(_managed(runs_held=999), _pos(cur=0.80))
    assert res.plan == []


def test_rank_dropout_is_never_evaluated_here():
    policy = ExecPolicy(stop_loss=0.50, grace_bars=1, trail_stop=None)
    res = _eval(_managed(bars_out=99), _pos(cur=0.80), policy=policy)
    assert res.plan == []


# --- cadence-sensitive rules stay opt-in --------------------------------------

def test_trailing_stop_is_off_by_default():
    """trail_stop=0.35 was calibrated on 4H sampling; do not apply it silently."""
    # +100% peak, now +20% -> a 35% giveback from the peak.
    res = _eval(_managed(peak_gain=1.0), _pos(cur=1.10))
    assert res.plan == []

    on = _eval(_managed(peak_gain=1.0), _pos(cur=1.10),
               cfg=RiskPassConfig(trailing_stop=True))
    assert on.plan and on.plan[0][3] == "trail_-35%"


def test_peak_is_not_ratcheted_while_the_trail_is_off():
    """Recording intraday peaks would tighten the 4H runner's trail unasked."""
    res = _eval(_managed(peak_gain=0.10), _pos(cur=1.84))  # +100% right now
    assert res.new_managed["HPE"]["peak_gain"] == 0.10

    on = _eval(_managed(peak_gain=0.10), _pos(cur=1.84),
               cfg=RiskPassConfig(trailing_stop=True))
    assert on.new_managed["HPE"]["peak_gain"] == 1.0


def test_take_profit_trim_is_off_by_default():
    res = _eval(_managed(), _pos(cur=1.20))  # +30% against a 20% take-profit
    assert res.plan == []

    on = _eval(_managed(), _pos(cur=1.20), cfg=RiskPassConfig(take_profit_trim=True))
    assert on.plan and on.plan[0][3] == "take_profit_+20%"
    assert on.plan[0][2] == 26  # scale_frac 0.5 of 52


# --- expiry flatten: the rule that would have saved HPE/ZM --------------------

def test_expiry_flatten_fires_on_the_last_tradable_session():
    now = dt.datetime(2026, 8, 14, 15, 50, tzinfo=ET)  # expiry day, past 15:45
    res = _eval(_managed(), _pos(cur=0.92), now=now)  # flat, so no stop
    assert res.plan == [(HPE, "sell", 52, "expiring_before_closure", "option")]


def test_expiry_flatten_waits_for_the_cutoff():
    now = dt.datetime(2026, 8, 14, 11, 0, tzinfo=ET)  # expiry day, before 15:45
    res = _eval(_managed(), _pos(cur=0.92), now=now)
    assert res.plan == []


def test_expiry_flatten_ignores_a_contract_with_time_left():
    now = dt.datetime(2026, 8, 12, 15, 50, tzinfo=ET)
    res = _eval(_managed(), _pos(cur=0.92), now=now)
    assert res.plan == []


def test_expiry_flatten_covers_the_session_before_a_weekend():
    """Friday is the last chance to sell a Saturday-stamped monthly on screen."""
    friday = dt.datetime(2026, 8, 14, 15, 50, tzinfo=ET)
    assert expiring_before_next_session("HPE260815C00060000", friday, RiskPassConfig())


def test_equity_positions_are_never_expiry_flattened():
    managed = {"AAPL": {"route": "equity", "symbol": "AAPL", "shares": 100,
                        "entry_avg_price": 100.0, "last_mark_price": 100.0}}
    pos = {"AAPL": {"qty": 100, "avg_entry": 100.0, "current": 99.0}}
    res = _eval(managed, pos, now=dt.datetime(2026, 8, 14, 15, 50, tzinfo=ET))
    assert res.plan == []


def test_parse_occ_expiry():
    assert parse_occ_expiry(HPE) == dt.date(2026, 8, 14)
    assert parse_occ_expiry("AAPL") is None


# --- safety interlocks --------------------------------------------------------

def test_a_resting_exit_order_is_never_double_sold():
    managed = _managed(exit_pending={"order_id": "x", "reason": "stop_-50%",
                                     "route": "option", "qty": 52.0,
                                     "entry_avg_price": 0.92})
    res = _eval(managed, _pos(cur=0.40))
    assert res.plan == []
    assert res.skipped["HPE"] == "exit_order_already_resting"


def test_mark_anomaly_blocks_the_stop():
    """A ~10x mark move is a corporate action, not a price. Never stop out on it.

    The real case: 2026-08-10, TENX marked 13.30 -> 1.42 with the share count
    unchanged (an unadjusted forward split). It tripped stop_-39% and was
    liquidated for -$1,457.
    """
    managed = {"TENX": {"route": "equity", "symbol": "TENX", "shares": 100,
                        "runs_held": 3, "bars_out": 0,
                        "entry_avg_price": 15.99, "last_mark_price": 13.30}}
    pos = {"TENX": {"qty": 100, "avg_entry": 15.99, "current": 1.42}}

    # Without the guard this is -91% against a 50% stop — an instant liquidation.
    assert risk_exit_action(1.42 / 15.99 - 1, trimmed=False, peak_gain=None,
                            policy=DEALER, cfg=RiskPassConfig())[0] == "exit"

    res = _eval(managed, pos)
    assert res.plan == []
    assert res.skipped["TENX"] == "mark_anomaly"
    assert res.anomalies["TENX"]["ratio"] > 9


def test_a_stale_option_mark_is_corroborated_before_stopping_out():
    """The live mid says the broker mark was stale — hold instead of stopping."""
    client = _Client(quote={"bid_price": 0.90, "ask_price": 0.94})
    res = _eval(_managed(), _pos(cur=0.40), client=client)
    assert res.plan == []
    assert res.anomalies["HPE"]["was"] == "stop_-50%"
    assert res.anomalies["HPE"]["now"] == "hold"


def test_module_state_lock_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr("core.live_risk_pass.LOCK_ROOT", tmp_path)
    with module_state_lock("dealer_ranker") as first:
        assert first is True
        with module_state_lock("dealer_ranker") as second:
            # The 4H runner owns the state; this tick must stand down.
            assert second is False
    with module_state_lock("dealer_ranker") as again:
        assert again is True


def test_risk_exit_action_matches_the_4h_reason_strings():
    """A stop fired here must be indistinguishable in the ledger from a 4H stop."""
    from core.live_4h_exec import exit_action

    gain, peak = -0.60, 0.0
    risk = risk_exit_action(gain, trimmed=False, peak_gain=peak, policy=DEALER,
                            cfg=RiskPassConfig())
    four_h = exit_action(gain, 1, 0, False, DEALER, peak_gain=peak)
    assert risk == four_h == ("exit", "stop_-50%")
