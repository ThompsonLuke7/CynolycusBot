from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.spy_intraday.Models.competition_ranker import (
    CompetitionSwingRanker,
    EmpiricalScoreCalibrator,
)


def test_empirical_score_calibrator_maps_to_percentiles() -> None:
    calibrator = EmpiricalScoreCalibrator.fit(np.array([1.0, 2.0, 3.0, 4.0]))
    actual = calibrator.transform(np.array([0.0, 2.0, 4.0, np.nan]))
    np.testing.assert_allclose(actual[:3], np.array([0.0, 0.5, 1.0]))
    assert np.isnan(actual[3])


def test_competition_ranker_loads_and_scores_causal_ohlcv() -> None:
    index = pd.date_range("2026-06-01 09:30", periods=120, freq="10min", tz="America/New_York")
    close = pd.Series(np.linspace(590.0, 598.0, len(index)), index=index)
    frame = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000.0,
            "ret_2": close.pct_change(2),
        },
        index=index,
    )
    ranker = CompetitionSwingRanker("Data/models/ga_xgboost/10min/competition_20260619")
    result = ranker.predict_frame(frame)
    assert list(result.columns) == ["short", "neutral", "long"]
    assert len(result) == len(frame)
    assert result[["short", "long"]].notna().all().all()
    assert result[["short", "long"]].ge(0.0).all().all()
    assert result[["short", "long"]].le(1.0).all().all()
