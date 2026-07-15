from __future__ import annotations

import json

import pandas as pd
import pytest

from strategies.dealer_positioning.scripts.build_dealer_rankings import build_rankings, run

pytestmark = pytest.mark.safe


def _row(
    symbol: str,
    scope: str,
    *,
    vacuum: float,
    density: float,
    pinning: float,
    imbalance: float,
    magnet: float,
    changes: dict | None = None,
) -> dict:
    row = {
        "symbol": symbol,
        "scope": scope,
        "snapshot_date": "2026-07-03",
        "captured_at": "2026-07-03T20:00:00Z",
        "vacuum_score": vacuum,
        "pinning_score": pinning,
        "gamma_density": density,
        "dealer_imbalance": imbalance,
        "pct_to_call_wall": 0.10 if symbol == "FAST" else 0.02,
        "pct_to_put_wall": -0.08 if symbol == "FAST" else -0.02,
        "pct_to_magnet": magnet,
        "distance_floor_ceiling": 0.20 if symbol == "FAST" else 0.04,
        "gex_concentration_index": 0.08 if symbol == "FAST" else 0.55,
    }
    if changes:
        row.update(changes)
    return row


def test_build_rankings_prefers_vacuum_sparse_gamma_and_room_to_move():
    rows = []
    for scope in ("daily_week", "through_month", "two_months"):
        rows.append(_row("FAST", scope, vacuum=0.20, density=1_000, pinning=4.0, imbalance=0.60, magnet=0.08))
        rows.append(_row("PINNED", scope, vacuum=0.02, density=100_000, pinning=0.2, imbalance=0.05, magnet=0.002))

    ranked = build_rankings(pd.DataFrame(rows))

    assert ranked.iloc[0]["symbol"] == "FAST"
    assert ranked.iloc[0]["dealer_swing_potential_score"] > ranked.iloc[1]["dealer_swing_potential_score"]
    assert ranked.iloc[0]["dealer_direction"] == "bullish"
    assert ranked.iloc[0]["dealer_swing_rank"] == 1
    assert ranked.iloc[0]["dealer_change_intensity_score"] == 0.0
    assert "dealer_change_intensity_rank" in ranked.columns
    assert "dealer_change_bullish_rank" in ranked.columns
    assert "dealer_change_bearish_rank" in ranked.columns
    assert json.loads(ranked.iloc[0]["scope_scores_json"])[0]["scope"] in {"daily_week", "through_month", "two_months"}


def test_build_rankings_adds_change_intensity_and_direction_ranks():
    rows = []
    flat = {
        "gamma_flip_change_1d": 0.0,
        "call_wall_change": 0.0,
        "put_wall_change": 0.0,
        "magnet_change": 0.0,
        "vega_wall_change": 0.0,
        "ceiling_change": 0.0,
        "floor_change": 0.0,
        "gamma_flip_velocity_3d": 0.0,
        "magnet_velocity_3d": 0.0,
        "callwall_velocity_3d": 0.0,
    }
    bullish = {
        "gamma_flip_change_1d": 6.0,
        "call_wall_change": 10.0,
        "put_wall_change": 4.0,
        "magnet_change": 8.0,
        "vega_wall_change": 5.0,
        "ceiling_change": 7.0,
        "floor_change": 3.0,
        "gamma_flip_velocity_3d": 1.2,
        "magnet_velocity_3d": 1.8,
        "callwall_velocity_3d": 2.0,
    }
    bearish = {key: -0.5 * value for key, value in bullish.items()}
    for scope in ("daily_week", "through_month", "two_months"):
        rows.append(_row("FLAT", scope, vacuum=0.20, density=1_000, pinning=4.0, imbalance=0.60, magnet=0.08, changes=flat))
        rows.append(_row("BULL", scope, vacuum=0.02, density=100_000, pinning=0.2, imbalance=0.05, magnet=0.002, changes=bullish))
        rows.append(_row("BEAR", scope, vacuum=0.02, density=100_000, pinning=0.2, imbalance=0.05, magnet=0.002, changes=bearish))

    ranked = build_rankings(pd.DataFrame(rows))
    by_symbol = ranked.set_index("symbol")

    assert by_symbol.loc["FLAT", "dealer_swing_rank"] == 1
    assert by_symbol.loc["BULL", "dealer_change_intensity_rank"] == 1
    assert by_symbol.loc["BULL", "dealer_change_bullish_rank"] == 1
    assert by_symbol.loc["BEAR", "dealer_change_bearish_rank"] == 1
    assert by_symbol.loc["BULL", "dealer_change_direction"] == "bullish"
    assert by_symbol.loc["BEAR", "dealer_change_direction"] == "bearish"
    assert by_symbol.loc["BULL", "dealer_change_intensity_score"] > by_symbol.loc["FLAT", "dealer_change_intensity_score"]
    assert json.loads(by_symbol.loc["BULL", "scope_scores_json"])[0]["scope_change_intensity_score"] > 0


def test_run_writes_daily_and_latest_ranking_files(tmp_path):
    snapshot_dir = tmp_path / "snapshots" / "20260703"
    snapshot_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            _row("FAST", "daily_week", vacuum=0.20, density=1_000, pinning=4.0, imbalance=0.60, magnet=0.08),
            _row("PINNED", "daily_week", vacuum=0.02, density=100_000, pinning=0.2, imbalance=0.05, magnet=0.002),
        ]
    )
    snapshot_path = snapshot_dir / "dealer_level_summary.parquet"
    frame.to_parquet(snapshot_path, index=False)

    result = run(snapshot_path=snapshot_path, output_root=tmp_path / "rankings")

    assert result.rows == 2
    assert result.output_path.exists()
    assert result.latest_path.exists()
    assert result.history_path.exists()
    out = pd.read_parquet(result.latest_path)
    assert list(out["symbol"]) == ["FAST", "PINNED"]
    history = pd.read_parquet(result.history_path)
    assert list(history["symbol"]) == ["FAST", "PINNED"]
    assert "ranking_source_path" in history.columns
