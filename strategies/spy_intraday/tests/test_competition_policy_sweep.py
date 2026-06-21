from __future__ import annotations

import pandas as pd

from core.API.Alpaca_API.inference.live_inference import LiveIndependentMetaXGBAgent
from scripts.sweep_live_thresholds_post_0401 import (
    PendingSetup,
    _regime_allowed,
    _triggered,
)
from strategies.spy_intraday.Policy.order_policy import OptionOrderPolicy, OptionOrderPolicyConfig


def _setup(side: str) -> PendingSetup:
    ts = pd.Timestamp("2026-06-01T10:00:00-04:00")
    return PendingSetup(
        side=side,
        ref=100.0,
        setup_ts=ts,
        setup_spot=99.5,
        start_ts=ts + pd.Timedelta(minutes=10),
        expires_at=ts + pd.Timedelta(minutes=20),
        prob=0.9,
        threshold=0.85,
        entry_kind="swing",
    )


def test_next_open_does_not_wait_for_breakout() -> None:
    row = pd.Series({"open": 99.5, "high": 99.8, "low": 99.2, "close": 99.6})
    assert _triggered(_setup("long"), row, mode="next_open")
    assert not _triggered(_setup("long"), row, mode="breakout")


def test_side_direction_is_respected_by_reversal_trigger() -> None:
    bullish = pd.Series({"open": 99.5, "high": 100.1, "low": 99.2, "close": 99.9})
    assert _triggered(_setup("long"), bullish, mode="reversal")
    assert not _triggered(_setup("short"), bullish, mode="reversal")


def test_one_minute_confirmation_variants() -> None:
    bullish_below_setup = pd.Series(
        {"open": 99.2, "high": 99.4, "low": 99.1, "close": 99.4}
    )
    bullish_above_setup = pd.Series(
        {"open": 99.6, "high": 100.1, "low": 99.5, "close": 99.9}
    )
    setup = _setup("long")
    assert _triggered(setup, bullish_below_setup, mode="body_1m")
    assert not _triggered(setup, bullish_below_setup, mode="reclaim_setup_close")
    assert _triggered(setup, bullish_above_setup, mode="reclaim_setup_close")

    setup = _setup("long")
    assert not _triggered(setup, bullish_below_setup, mode="body_2m")
    assert _triggered(setup, bullish_above_setup, mode="body_2m")


def test_regime_filter_groups() -> None:
    assert _regime_allowed("bullish", "bullish_neutral")
    assert _regime_allowed("neutral", "bullish_neutral")
    assert not _regime_allowed("bearish", "bullish_neutral")


def test_live_policy_can_override_only_the_long_trigger() -> None:
    policy = OptionOrderPolicy.__new__(OptionOrderPolicy)
    policy.cfg = OptionOrderPolicyConfig(
        meta_intrabar_entry_policy="phase4_swing_setup_bodyclose_bodyclose_v1",
        meta_intrabar_long_trigger_mode="next_open",
        meta_intrabar_short_trigger_mode="inherit",
    )
    assert policy._intrabar_trigger_mode(side="long") == "next_open"
    assert (
        policy._intrabar_trigger_mode(side="short")
        == "phase4_swing_setup_bodyclose_bodyclose_v1"
    )


def test_hybrid_source_uses_competition_long_and_active_short() -> None:
    index = pd.date_range("2026-06-01 09:30", periods=2, freq="10min", tz="America/New_York")
    base = pd.DataFrame({"close": [100.0, 101.0]}, index=index)

    class FakeArtifact:
        def __init__(self, long: float, short: float) -> None:
            self.long = long
            self.short = short

        def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "short": self.short,
                    "neutral": 0.1,
                    "long": self.long,
                },
                index=frame.index,
            )

    agent = LiveIndependentMetaXGBAgent.__new__(LiveIndependentMetaXGBAgent)
    agent._entry_prob_source = "competition_long_active_short"
    agent._competition_swing = FakeArtifact(long=0.91, short=0.12)
    agent._swing_setup_single = FakeArtifact(long=0.20, short=0.73)
    agent._swing_setup_probs_frame = None
    agent._swing_setup_annotated_cache = None
    agent._swing_setup_annotated_cache_key = None
    agent._build_setup_feature_frame = lambda **_: base

    result = agent._annotate_swing_setup_probs(df_1m=pd.DataFrame(), base_frame=base)
    assert result["p_swing_setup_long"].eq(0.91).all()
    assert result["p_swing_setup_short"].eq(0.73).all()
