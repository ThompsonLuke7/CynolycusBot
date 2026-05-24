from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DAILY_BARS_PATH,
    LEADERS_PER_THEME,
    THEME_LEADERS_PATH,
    THEME_SCORES_PATH,
    ensure_dirs,
    load_theme_memberships,
)


def load_stock_theme_panel() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    bars = bars.sort_values(["ticker", "date"])
    g = bars.groupby("ticker", group_keys=False)
    bars["stock_return_1d"] = g["px"].pct_change()
    bars["stock_return_5d"] = g["px"].pct_change(5)
    bars["stock_rvol"] = bars["volume"] / g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    bars["sma20"] = g["px"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    bars["stock_above_20d"] = (bars["px"] > bars["sma20"]).astype(float)
    high_20d = g["px"].transform(lambda s: s.rolling(20, min_periods=10).max())
    high_63d = g["px"].transform(lambda s: s.rolling(63, min_periods=20).max())
    bars["stock_new_high"] = bars["px"] >= high_63d
    bars["breakout_quality"] = bars["px"] / high_20d - 1.0

    long_map = load_theme_memberships()
    return bars.merge(long_map[["ticker", "theme"]], on="ticker", how="inner")


def rank_pct(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(["date", "theme"])[column].rank(pct=True, ascending=True)


def add_leader_follower_scores(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["relative_strength_5d"] = panel["stock_return_5d"] - panel["theme_return_5d"]
    panel["return_5d_rank"] = rank_pct(panel, "stock_return_5d")
    panel["rvol_rank"] = rank_pct(panel, "stock_rvol")
    panel["relative_strength_rank"] = rank_pct(panel, "relative_strength_5d")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank stock leaders inside the strongest themes.")
    parser.add_argument("--top-themes", type=int, default=0, help="Themes per day to score. Use 0 for all themes.")
    parser.add_argument("--leaders-per-theme", type=int, default=LEADERS_PER_THEME)
    args = parser.parse_args()

    ensure_dirs()
    panel = load_stock_theme_panel()
    scores = pd.read_parquet(THEME_SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    if args.top_themes and args.top_themes > 0:
        top_theme_days = scores[scores["theme_rank"] <= args.top_themes][["date", "theme", "theme_return_5d"]]
    else:
        top_theme_days = scores[["date", "theme", "theme_return_5d"]]
    panel = panel.merge(top_theme_days, on=["date", "theme"], how="inner")
    panel["stock_vs_theme_5d"] = panel["stock_return_5d"] - panel["theme_return_5d"]
    panel = add_leader_follower_scores(panel)
    panel["leader_rank"] = panel["ticker_rank_in_theme"]
    out = panel[panel["leader_rank"] <= args.leaders_per_theme][
        [
            "date",
            "theme",
            "ticker",
            "leader_score",
            "leader_rank",
            "ticker_rank_in_theme",
            "theme_leader_ticker",
            "theme_leader_score",
            "theme_leader_return_1d",
            "theme_leader_return_5d",
            "ticker_lag_vs_theme_leader",
            "ticker_lag_vs_leader",
            "stock_return_1d",
            "stock_return_5d",
            "stock_vs_theme_5d",
            "stock_rvol",
            "return_5d_rank",
            "rvol_rank",
            "relative_strength_5d",
            "relative_strength_rank",
            "new_high_flag",
            "stock_new_high",
            "breakout_quality",
            "stock_above_20d",
        ]
    ].rename(columns={"stock_rvol": "rvol"})
    out = out.replace([np.inf, -np.inf], np.nan).sort_values(["date", "theme", "leader_rank"])
    out.to_parquet(THEME_LEADERS_PATH, index=False)
    print(f"saved {len(out):,} leader rows -> {THEME_LEADERS_PATH}")


if __name__ == "__main__":
    main()
