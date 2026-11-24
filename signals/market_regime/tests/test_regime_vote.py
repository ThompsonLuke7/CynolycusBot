"""The seven-signal regime vote: mapping, missing-input safety, and row/frame parity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.nervous_system.contracts.enums import MarketRegime
from signals.market_regime.regime_vote import (
    VOTE_INPUT_COLUMNS,
    classify_frame,
    classify_row,
    regime_for_votes,
)

CALM = {
    "qqq_fast_down": 0.0,
    "qqq_below_sma50": 0.0,
    "spy_trend_state": 1.0,
    "breadth_z": 0.4,
    "risk_appetite_z": 0.2,
    "credit_risk_z": 0.1,
    "spy_rv20_z": 0.0,
}
BEARISH = {
    "qqq_fast_down": 1.0,
    "qqq_below_sma50": 1.0,
    "spy_trend_state": -1.0,
    "breadth_z": -0.1,
    "risk_appetite_z": -0.1,
    "credit_risk_z": -0.1,
    "spy_rv20_z": 1.5,
}
ORDER = ("qqq_fast_down", "qqq_below_sma50", "spy_trend_state", "breadth_z",
         "risk_appetite_z", "credit_risk_z", "spy_rv20_z")


@pytest.mark.parametrize(
    ("votes", "expected"),
    [(0, MarketRegime.RISK_ON), (1, MarketRegime.RISK_ON), (2, MarketRegime.NEUTRAL),
     (3, MarketRegime.NEUTRAL), (4, MarketRegime.DETERIORATING), (5, MarketRegime.RISK_OFF),
     (6, MarketRegime.RISK_OFF), (7, MarketRegime.CRISIS)],
)
def test_vote_count_mapping(votes: int, expected: MarketRegime) -> None:
    assert regime_for_votes(votes) is expected
    row = {col: (BEARISH[col] if i < votes else CALM[col]) for i, col in enumerate(ORDER)}
    result = classify_row(row)
    assert result.votes == votes
    assert result.regime is expected


def test_regime_for_votes_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        regime_for_votes(8)


def test_thresholds_are_strict_at_the_boundary() -> None:
    # z exactly 0 is not bearish; rv z exactly 1 is not bearish; trend 0 is not bearish.
    row = dict(CALM, breadth_z=0.0, risk_appetite_z=0.0, credit_risk_z=0.0, spy_rv20_z=1.0, spy_trend_state=0.0)
    assert classify_row(row).votes == 0


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, "x"])
def test_any_missing_or_invalid_input_is_unknown_never_partial(bad: object) -> None:
    row = dict(BEARISH, credit_risk_z=bad)
    result = classify_row(row)
    assert result.regime is MarketRegime.UNKNOWN
    assert result.votes is None
    assert result.missing == ("credit",)
    assert result.metrics() == {}
    assert result.reason_codes == ("MARKET_REGIME_UNCLASSIFIED_RULE_VECTOR", "MARKET_REGIME_VOTE_MISSING_CREDIT")


def test_absent_column_is_unknown() -> None:
    row = dict(BEARISH)
    row.pop("qqq_fast_down")
    assert classify_row(row).regime is MarketRegime.UNKNOWN


def test_reason_codes_and_metrics_name_every_bearish_vote() -> None:
    result = classify_row(dict(CALM, breadth_z=-1.0, spy_rv20_z=2.0))
    assert result.regime is MarketRegime.NEUTRAL
    assert result.reason_codes == ("MARKET_REGIME_VOTE_2_OF_7", "MARKET_REGIME_VOTE_BEARISH_BREADTH",
                                   "MARKET_REGIME_VOTE_BEARISH_VOL")
    m = result.metrics()
    assert m["regime_vote_count"] == 2.0
    assert m["regime_vote_breadth"] == 1.0 and m["regime_vote_vol"] == 1.0
    assert m["regime_vote_credit"] == 0.0


def test_classify_frame_matches_classify_row() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "qqq_fast_down": rng.integers(0, 2, 50).astype(float),
        "qqq_below_sma50": rng.integers(0, 2, 50).astype(float),
        "spy_trend_state": rng.choice([-1.0, 1.0], 50),
        "breadth_z": rng.normal(size=50),
        "risk_appetite_z": rng.normal(size=50),
        "credit_risk_z": rng.normal(size=50),
        "spy_rv20_z": rng.normal(size=50),
    })
    frame.loc[3, "breadth_z"] = np.nan
    out = classify_frame(frame)
    for i, row in frame.iterrows():
        r = classify_row(row.to_dict())
        assert out.at[i, "regime"] == r.regime.value
        if r.votes is None:
            assert np.isnan(out.at[i, "regime_vote_count"])
        else:
            assert out.at[i, "regime_vote_count"] == r.votes
    assert out.at[3, "regime"] == "UNKNOWN"


def test_classify_frame_requires_every_input_column() -> None:
    with pytest.raises(KeyError):
        classify_frame(pd.DataFrame({c: [0.0] for c in VOTE_INPUT_COLUMNS[:-1]}))
