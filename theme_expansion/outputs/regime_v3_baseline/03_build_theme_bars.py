from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DAILY_BARS_PATH, THEME_DAILY_PATH, UNIVERSE_FILTER_PATH, ensure_dirs, load_theme_memberships


def load_constituent_panel() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["ticker", "date"])
    bars["close_for_return"] = bars["adj_close"].fillna(bars["close"])
    g = bars.groupby("ticker", group_keys=False)
    bars["stock_return_1d"] = g["close_for_return"].pct_change()
    bars["stock_return_3d"] = g["close_for_return"].pct_change(3)
    bars["stock_return_5d"] = g["close_for_return"].pct_change(5)
    bars["stock_return_10d"] = g["close_for_return"].pct_change(10)
    bars["stock_dollar_volume"] = bars["close_for_return"] * bars["volume"]
    bars["stock_sma20"] = g["close_for_return"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    bars["stock_sma50"] = g["close_for_return"].transform(lambda s: s.rolling(50, min_periods=25).mean())
    bars["stock_high_63d"] = g["close_for_return"].transform(lambda s: s.rolling(63, min_periods=20).max())
    bars["stock_above_20d"] = bars["close_for_return"] > bars["stock_sma20"]
    bars["stock_above_50d"] = bars["close_for_return"] > bars["stock_sma50"]
    bars["stock_new_high"] = bars["close_for_return"] >= bars["stock_high_63d"]

    long_map = load_theme_memberships()
    if UNIVERSE_FILTER_PATH.exists():
        eligible = pd.read_csv(UNIVERSE_FILTER_PATH)
        eligible["ticker"] = eligible["ticker"].astype(str).str.upper()
        eligible_tickers = set(eligible[eligible["is_eligible"].astype(bool)]["ticker"])
        bars = bars[bars["ticker"].isin(eligible_tickers)]
        long_map = long_map[long_map["ticker"].isin(eligible_tickers)]
    keep = [
        "ticker",
        "theme",
        "asset_type",
        "is_tradable",
        "is_watchlist_only",
        "theme_constituent_count",
    ]
    return bars.merge(long_map[[c for c in keep if c in long_map.columns]], on="ticker", how="inner")


def concentration_top3(dollar_volume: pd.Series) -> float:
    total = dollar_volume.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return dollar_volume.nlargest(3).sum() / total


def entropy_from_contribution(contribution: pd.Series) -> float:
    total = contribution.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    weights = contribution[contribution > 0] / total
    return float(-(weights * np.log(weights)).sum())


def leader_concentration(contribution: pd.Series) -> float:
    total = contribution.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float(contribution.max() / total)


def build_theme_daily(panel: pd.DataFrame) -> pd.DataFrame:
    panel["stock_abs_contribution"] = panel["stock_return_1d"].abs().fillna(0.0) * panel["stock_dollar_volume"].fillna(0.0)
    grouped = panel.groupby(["date", "theme"], sort=True)
    daily = grouped.agg(
        theme_return_1d=("stock_return_1d", "mean"),
        theme_return_3d=("stock_return_3d", "mean"),
        theme_return_5d=("stock_return_5d", "mean"),
        theme_return_10d=("stock_return_10d", "mean"),
        theme_volume=("volume", "sum"),
        theme_dollar_volume=("stock_dollar_volume", "sum"),
        theme_advancers_pct=("stock_return_1d", lambda s: float((s > 0).mean())),
        theme_new_high_pct=("stock_new_high", "mean"),
        theme_above_20d_pct=("stock_above_20d", "mean"),
        theme_above_50d_pct=("stock_above_50d", "mean"),
        theme_concentration_top3=("stock_dollar_volume", concentration_top3),
        theme_num_constituents=("ticker", "nunique"),
        is_tradable=("is_tradable", "max"),
        is_watchlist_only=("is_watchlist_only", "max"),
        theme_effective_constituents=("theme_constituent_count", "max"),
        leader_concentration=("stock_abs_contribution", leader_concentration),
        entropy_score=("stock_abs_contribution", entropy_from_contribution),
        leader_return_5d=("stock_return_5d", "max"),
    ).reset_index()

    daily = daily.sort_values(["theme", "date"])
    g = daily.groupby("theme", group_keys=False)
    daily["theme_index"] = g["theme_return_1d"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    daily["theme_rvol"] = daily["theme_volume"] / g["theme_volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    daily["leader_gap"] = daily["leader_return_5d"] - daily["theme_return_5d"]
    daily["theme_effective_breadth"] = np.exp(daily["entropy_score"])
    return daily.sort_values(["date", "theme"]).reset_index(drop=True)


def main() -> None:
    argparse.ArgumentParser(description="Aggregate stock bars into theme-level daily bars.").parse_args()
    ensure_dirs()
    panel = load_constituent_panel()
    daily = build_theme_daily(panel)
    daily.to_parquet(THEME_DAILY_PATH, index=False)
    print(f"saved {len(daily):,} theme-day rows -> {THEME_DAILY_PATH}")


if __name__ == "__main__":
    main()
