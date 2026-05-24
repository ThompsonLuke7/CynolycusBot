"""Fixture smoke tests for context backtest inputs."""

from __future__ import annotations

import pandas as pd
from unittest.mock import patch

from meta_context.backtest_inputs import (
    build_context_backtest_timestamps,
    build_context_backtest_universe,
    build_forward_performance_labels,
)


def test_backtest_timestamps_and_forward_labels(tmp_path) -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T14:30:00Z", periods=6, freq="30min"),
            "ticker": ["RKLB"] * 6,
            "close": [10, 11, 12, 13, 12, 15],
        }
    )
    timestamps = build_context_backtest_timestamps(bars, output_path=tmp_path / "timestamps.parquet")
    labels = build_forward_performance_labels(bars, output_path=tmp_path / "labels.parquet")
    assert len(timestamps) == 6
    assert round(float(labels.iloc[0]["forward_1bar_return"]), 6) == 0.1
    assert labels.iloc[0]["max_forward_return"] > 0
    assert labels.iloc[0]["expansion_label"] == 1.0


def test_combined_swing_and_momentum_universe(tmp_path) -> None:
    swing_csv = tmp_path / "swing.csv"
    pd.DataFrame(
        [
            {"ticker": "RKLB", "sector": "Industrials", "market_cap_bucket": "Mid", "type": "Stock"},
            {"ticker": "SPY", "sector": "Broad Index", "market_cap_bucket": "N/A", "type": "ETF"},
        ]
    ).to_csv(swing_csv, index=False)
    momentum = pd.DataFrame(
        [
            {"ticker": "RKLB", "sector": "Unknown", "market_cap_bucket": "Unknown", "type": "Stock", "notes": "momentum"},
            {"ticker": "IONQ", "sector": "Technology", "market_cap_bucket": "Mid", "type": "Stock", "notes": "momentum"},
        ]
    )
    with patch("momentum_expansion.data.universe.load_candidate_metadata", return_value=momentum):
        universe = build_context_backtest_universe(universe_csv=swing_csv, output_path=tmp_path / "universe.csv")
    assert universe["ticker"].tolist() == ["IONQ", "RKLB"]
    rklb = universe.loc[universe["ticker"].eq("RKLB")].iloc[0]
    assert bool(rklb["in_multi_ticker_swing"])
    assert bool(rklb["in_momentum_expansion"])
    assert rklb["universe_sources"] == "multi_ticker_swing|momentum_expansion"


def run_all() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:
        p = Path(d)
        test_backtest_timestamps_and_forward_labels(p)
        test_combined_swing_and_momentum_universe(p)
    print("meta_context smoke tests passed")


if __name__ == "__main__":
    run_all()
