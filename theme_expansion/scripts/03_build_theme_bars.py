from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DAILY_BARS_PATH,
    THEME_DAILY_PATH,
    THEME_MEMBER_GRAPH_PATH,
    UNIVERSE_FILTER_PATH,
    ensure_dirs,
    load_theme_memberships,
)


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
    bars["stock_rvol"] = bars["volume"] / g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
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
    panel = bars.merge(long_map[[c for c in keep if c in long_map.columns]], on="ticker", how="inner")
    if "is_tradable" not in panel.columns:
        panel["is_tradable"] = True
    if "is_watchlist_only" not in panel.columns:
        panel["is_watchlist_only"] = False
    if "theme_constituent_count" not in panel.columns:
        panel["theme_constituent_count"] = panel.groupby("theme")["ticker"].transform("nunique")
    return panel


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


def _theme_rank_pct(panel: pd.DataFrame, column: str) -> pd.Series:
    return panel.groupby(["date", "theme"])[column].rank(pct=True, ascending=True)


def add_leader_follower_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["theme_return_5d_for_rs"] = panel.groupby(["date", "theme"])["stock_return_5d"].transform("mean")
    panel["relative_strength_5d"] = panel["stock_return_5d"] - panel["theme_return_5d_for_rs"]
    panel["return_5d_rank"] = _theme_rank_pct(panel, "stock_return_5d")
    panel["rvol_rank"] = _theme_rank_pct(panel, "stock_rvol")
    panel["relative_strength_rank"] = _theme_rank_pct(panel, "relative_strength_5d")
    panel["new_high_flag"] = panel["stock_new_high"].fillna(False).astype(float)
    panel["leader_score"] = (
        0.4 * panel["return_5d_rank"].fillna(0.0)
        + 0.3 * panel["rvol_rank"].fillna(0.0)
        + 0.2 * panel["relative_strength_rank"].fillna(0.0)
        + 0.1 * panel["new_high_flag"].fillna(0.0)
    )
    panel["ticker_rank_in_theme"] = panel.groupby(["date", "theme"])["leader_score"].rank(
        ascending=False,
        method="first",
    )
    leader_rows = panel[panel["ticker_rank_in_theme"].eq(1.0)][
        ["date", "theme", "ticker", "stock_return_1d", "stock_return_5d", "leader_score"]
    ].rename(
        columns={
            "ticker": "theme_leader_ticker",
            "stock_return_1d": "theme_leader_return_1d",
            "stock_return_5d": "theme_leader_return_5d",
            "leader_score": "theme_leader_score",
        }
    )
    panel = panel.merge(leader_rows, on=["date", "theme"], how="left")
    panel["ticker_lag_vs_theme_leader"] = panel["theme_leader_return_5d"] - panel["stock_return_5d"]
    panel["ticker_lag_vs_leader"] = panel["ticker_lag_vs_theme_leader"]
    return panel


def build_theme_member_graph(panel: pd.DataFrame) -> pd.DataFrame:
    if "leader_score" not in panel.columns:
        panel = add_leader_follower_features(panel)
    keep = [
        "date",
        "theme",
        "ticker",
        "ticker_rank_in_theme",
        "leader_score",
        "theme_leader_ticker",
        "theme_leader_score",
        "theme_leader_return_1d",
        "theme_leader_return_5d",
        "ticker_lag_vs_theme_leader",
        "ticker_lag_vs_leader",
        "stock_return_1d",
        "stock_return_5d",
        "stock_rvol",
        "return_5d_rank",
        "rvol_rank",
        "relative_strength_5d",
        "relative_strength_rank",
        "new_high_flag",
        "stock_new_high",
        "stock_above_20d",
        "stock_above_50d",
        "is_tradable",
        "is_watchlist_only",
    ]
    graph = panel[[c for c in keep if c in panel.columns]].rename(columns={"stock_rvol": "rvol"})
    return graph.replace([np.inf, -np.inf], np.nan).sort_values(["date", "theme", "ticker_rank_in_theme", "ticker"])


def build_theme_daily(panel: pd.DataFrame) -> pd.DataFrame:
    if "leader_score" not in panel.columns:
        panel = add_leader_follower_features(panel)
    panel["stock_abs_contribution"] = panel["stock_return_1d"].abs().fillna(0.0) * panel["stock_dollar_volume"].fillna(0.0)
    grouped = panel.groupby(["date", "theme"], sort=True)
    daily = grouped.agg(
        theme_return_1d=("stock_return_1d", "mean"),
        theme_return_3d=("stock_return_3d", "mean"),
        theme_return_5d=("stock_return_5d", "mean"),
        theme_return_10d=("stock_return_10d", "mean"),
        theme_volume=("volume", "sum"),
        theme_dollar_volume=("stock_dollar_volume", "sum"),
        theme_avg_rvol=("stock_rvol", "mean"),
        theme_advancers_pct=("stock_return_1d", lambda s: float((s > 0).mean())),
        theme_new_high_pct=("stock_new_high", "mean"),
        theme_new_high_count=("stock_new_high", "sum"),
        theme_above_20d_pct=("stock_above_20d", "mean"),
        theme_above_50d_pct=("stock_above_50d", "mean"),
        theme_concentration_top3=("stock_dollar_volume", concentration_top3),
        theme_num_constituents=("ticker", "nunique"),
        is_tradable=("is_tradable", "max"),
        is_watchlist_only=("is_watchlist_only", "max"),
        theme_effective_constituents=("theme_constituent_count", "max"),
        leader_concentration=("stock_abs_contribution", leader_concentration),
        entropy_score=("stock_abs_contribution", entropy_from_contribution),
        theme_leader_ticker=("theme_leader_ticker", "first"),
        theme_leader_score=("theme_leader_score", "first"),
        theme_leader_return_1d=("theme_leader_return_1d", "first"),
        theme_leader_return_5d=("theme_leader_return_5d", "first"),
        leader_return_5d=("theme_leader_return_5d", "first"),
        theme_avg_lag_vs_leader=("ticker_lag_vs_theme_leader", "mean"),
        theme_max_lag_vs_leader=("ticker_lag_vs_theme_leader", "max"),
    ).reset_index()

    daily = daily.sort_values(["theme", "date"])
    g = daily.groupby("theme", group_keys=False)
    daily["theme_index"] = g["theme_return_1d"].transform(lambda s: (1.0 + s.fillna(0.0)).cumprod())
    daily["theme_rvol"] = daily["theme_volume"] / g["theme_volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    daily["theme_breadth_20ema"] = g["theme_advancers_pct"].transform(
        lambda s: s.ewm(span=20, min_periods=5, adjust=False).mean()
    )
    daily["leader_gap"] = daily["leader_return_5d"] - daily["theme_return_5d"]
    daily["theme_effective_breadth"] = np.exp(daily["entropy_score"])
    return daily.sort_values(["date", "theme"]).reset_index(drop=True)


def main() -> None:
    argparse.ArgumentParser(description="Aggregate stock bars into theme-level daily bars.").parse_args()
    ensure_dirs()
    panel = load_constituent_panel()
    panel = add_leader_follower_features(panel)
    graph = build_theme_member_graph(panel)
    daily = build_theme_daily(panel)
    graph.to_parquet(THEME_MEMBER_GRAPH_PATH, index=False)
    daily.to_parquet(THEME_DAILY_PATH, index=False)
    print(f"saved {len(graph):,} member graph rows -> {THEME_MEMBER_GRAPH_PATH}")
    print(f"saved {len(daily):,} theme-day rows -> {THEME_DAILY_PATH}")


if __name__ == "__main__":
    main()
