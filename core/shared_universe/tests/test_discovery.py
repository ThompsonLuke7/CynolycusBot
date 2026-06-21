"""Offline tests for the dynamic discovery + promotion gate (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from core.shared_universe import discovery as disc
from core.shared_universe.discovery import (
    DiscoveryConfig,
    cap_tier,
    compute_daily_metrics,
    evaluate_promotions,
    is_stock_only,
    load_pending,
    merge_pending,
    normalize_exchange,
    save_pending,
    save_novel_pending_view,
    screen_filters,
)

pytestmark = pytest.mark.safe

CFG = DiscoveryConfig()


def _bars(closes, volumes):
    return pd.DataFrame({"close": closes, "volume": volumes})


def test_cap_tier_buckets():
    assert cap_tier(500e9) == "mega"
    assert cap_tier(50e9) == "large"
    assert cap_tier(5e9) == "mid"
    assert cap_tier(500e6) == "small"
    assert cap_tier(120e6) == "micro"
    assert cap_tier(20e6) == "nano"
    assert cap_tier(None) == "unknown"
    assert cap_tier(float("nan")) == "unknown"


def test_stock_only_below_options_floor():
    # Sub-$1B -> route stock-only; >= $1B -> options allowed.
    assert is_stock_only(400e6, CFG) is True
    assert is_stock_only(5e9, CFG) is False
    assert is_stock_only(None, CFG) is True


def test_compute_daily_metrics_basic():
    closes = [10.0] * 25
    volumes = [1_000_000] * 25
    m = compute_daily_metrics(_bars(closes, volumes), adv_window=20)
    assert m["last_price"] == 10.0
    assert m["avg_dollar_volume_20d"] == pytest.approx(10_000_000.0)
    assert m["history_days"] == 25


def test_compute_daily_metrics_empty():
    m = compute_daily_metrics(pd.DataFrame(), adv_window=20)
    assert m["history_days"] == 0


def test_screen_filters_pass_and_fail():
    passing = screen_filters(
        {"last_price": 12.0, "avg_dollar_volume_20d": 8e6, "market_cap": 250e6, "history_days": 300},
        CFG,
    )
    assert passing["meets_all"] is True

    # $150M cap passes the $100M floor; too-thin ADV fails.
    thin = screen_filters(
        {"last_price": 12.0, "avg_dollar_volume_20d": 1e6, "market_cap": 150e6, "history_days": 300},
        CFG,
    )
    assert thin["passes_market_cap"] is True
    assert thin["passes_adv"] is False
    assert thin["meets_all"] is False

    # Missing market cap is a soft fail.
    no_cap = screen_filters(
        {"last_price": 12.0, "avg_dollar_volume_20d": 8e6, "market_cap": float("nan"), "history_days": 300},
        CFG,
    )
    assert no_cap["passes_market_cap"] is False

    nan_history = screen_filters(
        {"last_price": 12.0, "avg_dollar_volume_20d": 8e6, "market_cap": 150e6, "history_days": float("nan")},
        CFG,
    )
    assert nan_history["passes_history"] is False
    assert nan_history["meets_all"] is False


def test_normalize_exchange_handles_alpaca_enum():
    from alpaca.trading.enums import AssetExchange

    assert normalize_exchange(AssetExchange.NASDAQ) == "NASDAQ"
    assert normalize_exchange(AssetExchange.NYSEARCA) == "ARCA"


def test_merge_pending_upsert_and_source_union():
    existing = pd.DataFrame(
        [{"ticker": "ABCD", "source": "catalyst", "status": "pending", "first_seen": "2026-06-01", "last_seen": "2026-06-01"}]
    )
    discovered = pd.DataFrame([{"ticker": "ABCD", "source": "screener", "last_price": 9.0}])
    merged = merge_pending(existing, discovered, today="2026-06-10")
    row = merged[merged["ticker"] == "ABCD"].iloc[0]
    assert row["first_seen"] == "2026-06-01"   # preserved
    assert row["last_seen"] == "2026-06-10"    # refreshed
    assert row["source"] == "both"             # unioned
    assert float(row["last_price"]) == 9.0


def test_save_novel_pending_view_excludes_known_and_promoted(tmp_path):
    pending = pd.DataFrame(
        [
            {"ticker": "KNOWN", "status": "pending"},
            {"ticker": "NOVEL", "status": "pending"},
            {"ticker": "DONE", "status": "promoted"},
        ]
    )
    path = tmp_path / "pending_tickers_novel.csv"

    save_novel_pending_view(pending, known_tickers=["KNOWN"], path=path)

    saved = pd.read_csv(path)
    assert saved["ticker"].tolist() == ["NOVEL"]


def test_promotion_gate_releases_when_metrics_met():
    pending = pd.DataFrame(
        [
            {"ticker": "GOOD", "status": "pending", "last_price": 15.0, "avg_dollar_volume_20d": 9e6, "market_cap": 300e6, "history_days": 260},
            {"ticker": "THIN", "status": "pending", "last_price": 15.0, "avg_dollar_volume_20d": 1e6, "market_cap": 300e6, "history_days": 260},
            {"ticker": "NEW", "status": "pending", "last_price": 15.0, "avg_dollar_volume_20d": 9e6, "market_cap": 300e6, "history_days": 40},
        ]
    )
    out = evaluate_promotions(pending, CFG, already_eligible=[], today="2026-06-10")
    by = out.set_index("ticker")
    assert by.loc["GOOD", "status"] == "promoted"
    assert by.loc["GOOD", "promoted_date"] == "2026-06-10"
    assert by.loc["THIN", "status"] == "pending"
    assert "adv" in by.loc["THIN", "reject_reason"]
    # Not enough history -> quarantined even though liquidity is fine.
    assert by.loc["NEW", "status"] == "pending"
    assert "history" in by.loc["NEW", "reject_reason"]

def test_promotion_gate_requires_multiple_observation_days():
    pending = pd.DataFrame(
        [
            {
                "ticker": "FRESH",
                "status": "pending",
                "first_seen": "2026-06-10",
                "last_price": 15.0,
                "avg_dollar_volume_20d": 9e6,
                "market_cap": 300e6,
                "history_days": 260,
            }
        ]
    )

    same_day = evaluate_promotions(pending, CFG, already_eligible=[], today="2026-06-10")
    matured = evaluate_promotions(pending, CFG, already_eligible=[], today="2026-06-12")

    assert same_day.iloc[0]["status"] == "pending"
    assert "observation" in same_day.iloc[0]["reject_reason"]
    assert matured.iloc[0]["status"] == "promoted"


def test_premature_same_day_promotion_is_repaired():
    pending = pd.DataFrame(
        [
            {
                "ticker": "PREMATURE",
                "status": "promoted",
                "first_seen": "2026-06-18",
                "promoted_date": "2026-06-18",
                "last_price": 15.0,
                "avg_dollar_volume_20d": 9e6,
                "market_cap": 300e6,
                "history_days": 260,
            }
        ]
    )

    out = evaluate_promotions(
        pending,
        CFG,
        already_eligible=["PREMATURE"],
        today="2026-06-18",
    )

    assert out.iloc[0]["status"] == "pending"
    assert "observation" in out.iloc[0]["reject_reason"]


def test_promotion_latches_and_respects_existing_universe():
    pending = pd.DataFrame(
        [
            {"ticker": "ALREADY", "status": "pending", "last_price": 1.0, "avg_dollar_volume_20d": 0.0, "market_cap": 0.0, "history_days": 0},
            {"ticker": "LATCHED", "status": "promoted", "last_price": 1.0, "avg_dollar_volume_20d": 0.0, "market_cap": 0.0, "history_days": 0, "promoted_date": "2026-05-01"},
        ]
    )
    out = evaluate_promotions(pending, CFG, already_eligible=["ALREADY"], today="2026-06-10")
    by = out.set_index("ticker")
    # In-universe name is marked promoted regardless of (stale) metrics.
    assert by.loc["ALREADY", "status"] == "promoted"
    # Previously promoted stays promoted and keeps its original date.
    assert by.loc["LATCHED", "status"] == "promoted"
    assert by.loc["LATCHED", "promoted_date"] == "2026-05-01"


def test_pending_roundtrip(tmp_path):
    path = tmp_path / "pending_tickers.csv"
    pending = merge_pending(pd.DataFrame(), pd.DataFrame([{"ticker": "WXYZ", "source": "screener"}]), today="2026-06-10")
    save_pending(pending, path)
    loaded = load_pending(path)
    assert "WXYZ" in set(loaded["ticker"])
    assert list(loaded.columns) == disc.PENDING_COLUMNS


def test_universe_promoted_names_become_eligible(tmp_path, monkeypatch):
    """A promoted pending row flows through to is_eligible in the built universe."""
    from core.shared_universe import universe as uni

    pending_path = tmp_path / "pending_tickers.csv"
    promoted = pd.DataFrame(
        [{"ticker": "ZZZZ", "status": "promoted", "cap_tier": "small", "stock_only": True}]
    )
    promoted.to_csv(pending_path, index=False)
    monkeypatch.setattr(uni, "PENDING_TICKERS_CSV", pending_path)

    out_path = tmp_path / "shared_universe.csv"
    built = uni.build_shared_universe(out_path=out_path)
    row = built[built["ticker"] == "ZZZZ"]
    assert not row.empty
    assert bool(row.iloc[0]["is_eligible"]) is True
    assert row.iloc[0]["eligible_reason"] == "dynamic_discovery"
