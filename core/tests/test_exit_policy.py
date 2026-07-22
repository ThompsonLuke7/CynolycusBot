"""Exit-policy ordering for the shared 4H engine.

Locks in the backtest-driven change: stop-loss > take-profit scale-out > horizon >
rank drop-out (drop-out OFF by default so positions ride to horizon). The old code
exited on drop-out first, which dumped winners early and rode rank-sticky losers
(the BE/COHR options) all the way down.
"""
from core.live_4h_exec import ExecPolicy, exit_action


def test_stoploss_takes_priority_and_catches_ride_to_zero():
    p = ExecPolicy(stop_loss=0.50)  # explicit: independent of the live default
    # a -55% option-premium loser exits at the stop even while still top-ranked
    assert exit_action(-0.55, runs_held=5, bars_out=0, trimmed=False, policy=p) == ("exit", "stop_-50%")


def test_takeprofit_scaleout_before_horizon():
    p = ExecPolicy(take_profit=0.20, horizon_bars=25)  # explicit: independent of the live default
    assert exit_action(0.25, runs_held=3, bars_out=0, trimmed=False, policy=p) == ("trim", "take_profit_+20%")
    # already trimmed winner keeps riding
    assert exit_action(0.25, runs_held=3, bars_out=0, trimmed=True, policy=p) == ("hold", "")


def test_dropout_disabled_by_default_rides_to_horizon():
    p = ExecPolicy(horizon_bars=25)  # grace_bars=None default; horizon explicit for this test's threshold
    # flat name out of top-K for 10 bars is HELD (old code would have dropped it)
    assert exit_action(0.01, runs_held=8, bars_out=10, trimmed=False, policy=p) == ("hold", "")
    # ...until the horizon hard cap
    assert exit_action(0.01, runs_held=25, bars_out=10, trimmed=False, policy=p) == ("exit", "horizon")


def test_dropout_backstop_still_works_when_grace_set():
    p = ExecPolicy(grace_bars=3)
    assert exit_action(0.01, runs_held=8, bars_out=5, trimmed=False, policy=p) == ("exit", "dropped_out")


def test_stoploss_disabled_when_zero_or_none():
    for sl in (0.0, None):
        p = ExecPolicy(stop_loss=sl)
        assert exit_action(-0.90, runs_held=3, bars_out=0, trimmed=False, policy=p) == ("hold", "")
