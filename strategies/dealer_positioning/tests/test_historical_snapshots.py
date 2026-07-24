from __future__ import annotations

import json
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from strategies.dealer_positioning.scripts.capture_historical_snapshots import (
    _load_prior_summary,
    _matrix_features,
    _summary_row,
    count_candidate_symbols,
    load_symbols,
)

pytestmark = pytest.mark.safe


def test_liquidity_prefilter_counts_large_adv_and_cap_proxy(tmp_path):
    universe = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "ticker": ["BIG", "FAST", "TINY", "SKIP"],
            "is_eligible": [True, True, True, False],
            "avg_dollar_volume_20d": [1_000_000, 60_000_000, 1_000_000, 100_000_000],
            "market_cap": [0, 0, 0, 0],
            "market_cap_bucket": ["Large", "Small", "Small", "Mega"],
            "cap_tier": ["large", "small", "small", "mega"],
            "stock_only": [False, False, False, False],
        }
    ).to_csv(universe, index=False)

    assert load_symbols(universe_path=universe) == ["BIG", "FAST"]
    counts = count_candidate_symbols(universe_path=universe)
    assert counts["eligible_rows"] == 3
    assert counts["candidate_rows"] == 2
    assert counts["adv_pass"] == 1
    assert counts["large_or_mega_proxy_pass"] == 1


def test_snapshot_summary_persists_requested_level_and_matrix_features():
    levels = {
        "spot": 100.0,
        "gamma_flip": 102.0,
        "call_wall": 110.0,
        "put_wall": 90.0,
        "nearest_magnet": 105.0,
        "next_vega_wall_above": 108.0,
        "next_vega_wall_below": 95.0,
        "vega_wall": 95.0,
        "total_gex": -20_000.0,
    }
    rows = [
        {"strike": 90.0, "spot": 100.0, "net_gex": -1000, "total_abs_gex": 1000, "call_gex": 0, "put_gex": -1000, "call_oi": 10, "put_oi": 200, "call_volume": 1, "put_volume": 20},
        {"strike": 100.0, "spot": 100.0, "net_gex": 500, "total_abs_gex": 500, "call_gex": 400, "put_gex": -100, "call_oi": 50, "put_oi": 10, "call_volume": 3, "put_volume": 1},
        {"strike": 110.0, "spot": 100.0, "net_gex": 2000, "total_abs_gex": 2000, "call_gex": 2000, "put_gex": 0, "call_oi": 300, "put_oi": 0, "call_volume": 30, "put_volume": 0},
    ]
    grid = {
        "symbol": "TST",
        "levels": levels,
        "rows": rows,
        "expirations": ["2026-07-10"],
        "available_expirations": ["2026-07-10"],
        "strike_window_pct": 0.5,
    }
    summary = _summary_row(
        grid=grid,
        levels=levels,
        scope="daily_week",
        captured_at=datetime(2026, 7, 3, tzinfo=ZoneInfo("America/New_York")),
        snapshot_date="2026-07-02",
        row_meta={"avg_dollar_volume_20d": 75_000_000, "market_cap_bucket": "Large"},
        liquidity={"total_option_oi": 570.0, "total_option_volume": 55.0, "liquidity_pass_reason": "adv"},
    )
    summary.update(_matrix_features(rows, levels=levels))

    assert summary["pct_to_gamma_flip"] == pytest.approx(0.02)
    assert summary["inside_call_wall_zone"] is True
    assert summary["floor"] == 90.0
    assert summary["ceiling"] == 110.0
    assert summary["largest_positive_gex"] == 2000.0
    assert summary["dealer_imbalance"] == pytest.approx(1500.0 / 3500.0)
    assert summary["wall_dominance"] == pytest.approx(2.0)
    assert summary["snapshot_date"] == "2026-07-02"
    assert json.loads(summary["expirations"]) == ["2026-07-10"]


def test_load_prior_summary_no_future_warning_with_an_empty_day(tmp_path):
    # 2026-07-13's summary has rows; 2026-07-14's is present but empty (e.g. a
    # capture that ran with zero eligible candidates) -- pandas warns on
    # concat unless empty/all-NA frames are excluded first.
    day1 = tmp_path / "20260713"
    day1.mkdir()
    pd.DataFrame({"symbol": ["AAA"], "scope": ["daily_week"], "captured_at": ["2026-07-13T20:00:00Z"]}).to_parquet(
        day1 / "dealer_level_summary.parquet", index=False
    )
    day2 = tmp_path / "20260714"
    day2.mkdir()
    pd.DataFrame(columns=["symbol", "scope", "captured_at"]).to_parquet(
        day2 / "dealer_level_summary.parquet", index=False
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _load_prior_summary(output_root=tmp_path, before_date="2026-07-15")

    concat_warnings = [w for w in caught if issubclass(w.category, FutureWarning) and "concat" in str(w.message)]
    assert concat_warnings == []
    assert list(out["symbol"]) == ["AAA"]


def test_load_prior_summary_missing_root_returns_empty_frame(tmp_path):
    out = _load_prior_summary(output_root=tmp_path / "missing", before_date="2026-07-15")
    assert out.empty
