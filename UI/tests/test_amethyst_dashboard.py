from __future__ import annotations

import json

import pandas as pd
import pytest

from UI.amethyst_dashboard import AmethystDashboardApp, _PAGE, _json_safe
from UI.dealer_positioning_dashboard import _select_expirations_for_scope

pytestmark = pytest.mark.safe


def test_amethyst_view_joins_local_daily_bars_with_fake_dealer_grid(tmp_path):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=3, tz="UTC"),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.5, 10.5, 11.5],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 1200, 1400],
        }
    ).to_parquet(bars_dir / "BOX.parquet", index=False)

    def fake_grid(symbol, expiration_scope, window_pct):
        assert symbol == "BOX"
        assert expiration_scope == "daily_week"
        assert window_pct == 0.08
        return {
            "expiration_scope": expiration_scope,
            "expirations": ["2026-07-10"],
            "levels": {
                "spot": 12.5,
                "call_wall": 14.0,
                "put_wall": 11.0,
                "nearest_magnet": 13.0,
            },
            "rows": [
                {
                    "strike": 13.0,
                    "net_gex": 250000.0,
                    "total_abs_gex": 300000.0,
                    "total_vex": 12000.0,
                    "call_oi": 100,
                    "put_oi": 50,
                    "tags": ["magnet"],
                    "zone": "magnet",
                }
            ],
        }

    app = AmethystDashboardApp(bars_root=bars_dir, grid_loader=fake_grid)
    payload = app.view(symbol="box", days=30, expiration_scope="daily_week", window_pct=0.08)

    assert payload["symbol"] == "BOX"
    assert payload["expiration_scope"] == "daily_week"
    assert len(payload["bars"]) == 3
    assert payload["levels"]["nearest_magnet"] == 13.0
    assert payload["rows"][0]["zone"] == "magnet"
    json.dumps(_json_safe(payload), allow_nan=False)


def test_amethyst_view_keeps_bars_when_grid_lookup_fails(tmp_path):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=1, tz="UTC"),
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.5],
            "volume": [1000],
        }
    ).to_parquet(bars_dir / "SPY.parquet", index=False)

    app = AmethystDashboardApp(
        bars_root=bars_dir,
        grid_loader=lambda symbol, dtes, window_pct: (_ for _ in ()).throw(RuntimeError("chain unavailable")),
    )

    payload = app.view(symbol="SPY")

    assert len(payload["bars"]) == 1
    assert payload["rows"] == []
    assert payload["grid"]["error"] == "chain unavailable"


def test_amethyst_page_has_wired_modes_and_no_dead_nav_labels():
    assert "data-mode=gex" in _PAGE
    assert "data-mode=vex" in _PAGE
    assert "data-mode=matrix" in _PAGE
    assert "data-scope=daily_week" in _PAGE
    assert "data-scope=through_month" in _PAGE
    assert "data-scope=two_months" in _PAGE
    assert "data-scope=next_expiration" not in _PAGE
    assert "data-scope=three_months" not in _PAGE
    assert "Next Weekly" not in _PAGE
    assert "3 Exp" not in _PAGE
    assert "id=dtes" not in _PAGE
    assert "value=.50" in _PAGE
    assert "Read Only" not in _PAGE
    assert "Dealer levels" not in _PAGE
    assert "resetView" in _PAGE
    assert "addEventListener('wheel'" in _PAGE
    assert "id=insight" in _PAGE
    assert "spreadLevels" in _PAGE
    assert "heatMetric" in _PAGE
    assert "heatStyle" in _PAGE
    assert "drawHoverPriceLine" in _PAGE
    assert "HOVER_PRICE" in _PAGE
    assert "function levelPrice" in _PAGE
    assert "Number(levels[s[0]])" not in _PAGE
    assert "background-color:rgba" in _PAGE
    assert "linear-gradient(90deg,rgba('+color+'" not in _PAGE
    assert "title=" in _PAGE
    assert "Spyglass" not in _PAGE
    assert "Watchlists" not in _PAGE


def test_expiration_scope_selection_handles_weeklies_dailies_and_monthlies():
    expirations = [
        "2026-07-06",
        "2026-07-07",
        "2026-07-10",
        "2026-07-17",
        "2026-07-24",
        "2026-07-31",
        "2026-08-07",
        "2026-08-14",
        "2026-08-21",
        "2026-09-18",
    ]
    ref_date = pd.Timestamp("2026-07-03").date()

    assert _select_expirations_for_scope(expirations, "next_expiration", ref_date=ref_date) == ("2026-07-06",)
    assert _select_expirations_for_scope(expirations, "daily_week", ref_date=ref_date) == (
        "2026-07-06",
        "2026-07-07",
        "2026-07-10",
    )
    assert _select_expirations_for_scope(expirations, "through_month", ref_date=ref_date) == (
        "2026-07-06",
        "2026-07-07",
        "2026-07-10",
        "2026-07-17",
    )
    assert _select_expirations_for_scope(expirations, "two_months", ref_date=ref_date) == (
        "2026-07-06",
        "2026-07-07",
        "2026-07-10",
        "2026-07-17",
        "2026-07-24",
        "2026-07-31",
        "2026-08-07",
        "2026-08-14",
        "2026-08-21",
    )


def test_expiration_scope_selection_matches_nbis_weekly_monthly_shape():
    expirations = ["2026-07-10", "2026-07-17", "2026-07-24", "2026-07-31", "2026-08-07", "2026-08-21", "2026-09-18"]
    ref_date = pd.Timestamp("2026-07-03").date()

    assert _select_expirations_for_scope(expirations, "next_expiration", ref_date=ref_date) == ("2026-07-10",)
    assert _select_expirations_for_scope(expirations, "daily_week", ref_date=ref_date) == ("2026-07-10",)
    assert _select_expirations_for_scope(expirations, "through_month", ref_date=ref_date) == (
        "2026-07-10",
        "2026-07-17",
    )
    assert _select_expirations_for_scope(expirations, "two_months", ref_date=ref_date) == (
        "2026-07-10",
        "2026-07-17",
        "2026-07-24",
        "2026-07-31",
        "2026-08-07",
        "2026-08-21",
    )
