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
    bars["stock_return_5d"] = g["px"].pct_change(5)
    bars["stock_rvol"] = bars["volume"] / g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    bars["sma20"] = g["px"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    bars["stock_above_20d"] = (bars["px"] > bars["sma20"]).astype(float)
    high_20d = g["px"].transform(lambda s: s.rolling(20, min_periods=10).max())
    bars["breakout_quality"] = bars["px"] / high_20d - 1.0

    long_map = load_theme_memberships()
    return bars.merge(long_map[["ticker", "theme"]], on="ticker", how="inner")


def rank_pct(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(["date", "theme"])[column].rank(pct=True, ascending=True)


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

    panel["leader_score"] = (
        0.40 * rank_pct(panel, "stock_vs_theme_5d").fillna(0.0)
        + 0.25 * rank_pct(panel, "stock_rvol").fillna(0.0)
        + 0.20 * rank_pct(panel, "breakout_quality").fillna(0.0)
        + 0.15 * rank_pct(panel, "stock_above_20d").fillna(0.0)
    )
    panel["leader_rank"] = panel.groupby(["date", "theme"])["leader_score"].rank(ascending=False, method="first")
    out = panel[panel["leader_rank"] <= args.leaders_per_theme][
        [
            "date",
            "theme",
            "ticker",
            "leader_score",
            "leader_rank",
            "stock_return_5d",
            "stock_vs_theme_5d",
            "stock_rvol",
            "breakout_quality",
            "stock_above_20d",
        ]
    ].rename(columns={"stock_rvol": "rvol"})
    out = out.replace([np.inf, -np.inf], np.nan).sort_values(["date", "theme", "leader_rank"])
    out.to_parquet(THEME_LEADERS_PATH, index=False)
    print(f"saved {len(out):,} leader rows -> {THEME_LEADERS_PATH}")


if __name__ == "__main__":
    main()
