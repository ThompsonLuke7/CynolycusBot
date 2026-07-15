from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from UI.dealer_ranker_dashboard import _cross_day_rows, _ranking_path_date, _top_rows

pytestmark = pytest.mark.safe


def test_ranking_path_date_ignores_latest_and_history():
    assert _ranking_path_date(Path("dealer_swing_rankings_20260709.parquet")) == "2026-07-09"
    assert _ranking_path_date(Path("dealer_swing_rankings_latest.parquet")) is None
    assert _ranking_path_date(Path("dealer_swing_rankings_history.parquet")) is None


def test_top_rows_returns_structural_and_shift_fields():
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "dealer_swing_rank": 2,
                "dealer_swing_potential_score": 80.0,
                "dealer_direction": "neutral",
                "dealer_change_intensity_rank": 1,
                "dealer_change_bullish_rank": 1,
                "dealer_change_bearish_rank": 2,
                "dealer_change_direction_bias": 0.5,
            },
            {
                "symbol": "BBB",
                "dealer_swing_rank": 1,
                "dealer_swing_potential_score": 90.0,
                "dealer_direction": "bullish",
                "dealer_change_intensity_rank": 2,
                "dealer_change_bullish_rank": 2,
                "dealer_change_bearish_rank": 1,
                "dealer_change_direction_bias": -0.5,
            },
        ]
    )

    rows = _top_rows(
        frame,
        rank_col="dealer_swing_rank",
        score_col="dealer_swing_potential_score",
        direction_col="dealer_direction",
        top=2,
    )

    assert [row["ticker"] for row in rows] == ["BBB", "AAA"]
    assert rows[0]["structural_rank"] == 1
    assert rows[0]["shift_rank"] == 2


def test_cross_day_rows_sorts_by_rank_improvement():
    current = pd.DataFrame(
        [
            {"symbol": "AAA", "dealer_swing_rank": 5, "dealer_swing_potential_score": 70.0, "dealer_change_intensity_rank": 3, "dealer_change_intensity_score": 80.0, "dealer_change_direction": "bullish", "dealer_change_direction_bias": 0.3},
            {"symbol": "BBB", "dealer_swing_rank": 2, "dealer_swing_potential_score": 90.0, "dealer_change_intensity_rank": 1, "dealer_change_intensity_score": 95.0, "dealer_change_direction": "bullish", "dealer_change_direction_bias": 0.8},
        ]
    )
    previous = pd.DataFrame(
        [
            {"symbol": "AAA", "dealer_swing_rank": 6, "dealer_swing_potential_score": 68.0, "dealer_change_intensity_rank": 4, "dealer_change_intensity_score": 70.0},
            {"symbol": "BBB", "dealer_swing_rank": 50, "dealer_swing_potential_score": 50.0, "dealer_change_intensity_rank": 20, "dealer_change_intensity_score": 40.0},
        ]
    )

    rows = _cross_day_rows(current, previous, top=2)

    assert rows[0]["ticker"] == "BBB"
    assert rows[0]["structural_rank_improvement"] == 48.0
    assert rows[0]["shift_rank_improvement"] == 19.0
