from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    BENCHMARK_TICKERS,
    DAILY_BARS_PATH,
    THEME_DAILY_PATH,
    THEME_SCORE_WEIGHTS,
    THEME_SCORES_PATH,
    ensure_dirs,
)


def benchmark_returns() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars[bars["ticker"].isin(BENCHMARK_TICKERS)].copy()
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    wide = bars.pivot(index="date", columns="ticker", values="px").sort_index()
    out = pd.DataFrame(index=wide.index)
    for ticker in BENCHMARK_TICKERS:
        if ticker in wide:
            lower = ticker.lower()
            out[f"{lower}_return_5d"] = wide[ticker].pct_change(5)
            out[f"{lower}_return_10d"] = wide[ticker].pct_change(10)
            out[f"{lower}_return_20d"] = wide[ticker].pct_change(20)
    return out.reset_index()


def pct_rank_by_date(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.Series:
    return df.groupby("date")[column].rank(pct=True, ascending=ascending)


def dense_rank_by_date(df: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series:
    return df.groupby("date")[column].rank(ascending=ascending, method="first")


def score_rank_change(rank: pd.Series, dates: pd.Series, themes: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "theme": themes, "momentum_rank": rank}).sort_values(["theme", "date"])
    frame["momentum_rank_change_5d"] = frame.groupby("theme")["momentum_rank"].diff(5) * -1.0
    return frame.sort_index()["momentum_rank_change_5d"]


def score_rank_stability(rank: pd.Series, dates: pd.Series, themes: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "theme": themes, "momentum_rank": rank}).sort_values(["theme", "date"])
    rolling_std = frame.groupby("theme")["momentum_rank"].transform(lambda s: s.rolling(5, min_periods=3).std())
    frame["rank_stability_5d"] = 1.0 / (rolling_std + 1.0)
    return frame.sort_index()["rank_stability_5d"]


def rolling_rank_stability(rank: pd.Series, dates: pd.Series, themes: pd.Series, window: int) -> pd.Series:
    frame = pd.DataFrame({"date": dates, "theme": themes, "rank": rank}).sort_values(["theme", "date"])
    rolling_std = frame.groupby("theme")["rank"].transform(lambda s: s.rolling(window, min_periods=max(3, window // 2)).std())
    frame[f"rank_stability_{window}d"] = 1.0 / (rolling_std + 1.0)
    return frame.sort_index()[f"rank_stability_{window}d"]


def main() -> None:
    argparse.ArgumentParser(description="Build rule-based theme rotation scores.").parse_args()
    ensure_dirs()
    df = pd.read_parquet(THEME_DAILY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["theme", "date"]).merge(benchmark_returns(), on="date", how="left")

    df["theme_vs_spy_5d"] = df["theme_return_5d"] - df.get("spy_return_5d", np.nan)
    df["theme_vs_spy_10d"] = df["theme_return_10d"] - df.get("spy_return_10d", np.nan)
    df["theme_return_20d"] = df.groupby("theme")["theme_index"].pct_change(20)
    df["theme_vs_spy_20d"] = df["theme_return_20d"] - df.get("spy_return_20d", np.nan)
    df["theme_vs_qqq_5d"] = df["theme_return_5d"] - df.get("qqq_return_5d", np.nan)
    df["theme_vs_qqq_10d"] = df["theme_return_10d"] - df.get("qqq_return_10d", np.nan)
    df["theme_vs_qqq_20d"] = df["theme_return_20d"] - df.get("qqq_return_20d", np.nan)
    df["theme_breadth"] = df["theme_advancers_pct"]
    df = df.sort_values(["theme", "date"])
    df["theme_breadth_10d"] = df.groupby("theme")["theme_breadth"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    df["theme_breadth_20d"] = df.groupby("theme")["theme_breadth"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["theme_rvol_10d"] = df.groupby("theme")["theme_rvol"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    if "is_tradable" not in df.columns:
        df["is_tradable"] = True
    if "is_watchlist_only" not in df.columns:
        df["is_watchlist_only"] = False

    df["rank_5d"] = pct_rank_by_date(df, "theme_vs_spy_5d", ascending=True)
    df["momentum_rank_5d_raw"] = dense_rank_by_date(df, "theme_vs_spy_5d", ascending=False)
    df["momentum_rank_change_5d_raw"] = score_rank_change(df["momentum_rank_5d_raw"], df["date"], df["theme"])
    df["rank_stability_5d"] = score_rank_stability(df["momentum_rank_5d_raw"], df["date"], df["theme"])
    df["rank_change_5d"] = df.groupby("date")["momentum_rank_change_5d_raw"].rank(pct=True, ascending=True)
    df["rank_stability_5d_rank"] = pct_rank_by_date(df, "rank_stability_5d", ascending=True)
    df["entropy_rank"] = pct_rank_by_date(df, "entropy_score", ascending=True)
    df["rvol_rank"] = pct_rank_by_date(df, "theme_rvol", ascending=True)
    df["breadth_rank"] = pct_rank_by_date(df, "theme_breadth", ascending=True)

    weight_total = sum(THEME_SCORE_WEIGHTS.values())
    score = (
        THEME_SCORE_WEIGHTS["rank_5d"] * df["rank_5d"].fillna(0.0)
        + THEME_SCORE_WEIGHTS["rank_change_5d"] * df["rank_change_5d"].fillna(0.5)
        + THEME_SCORE_WEIGHTS["theme_breadth"] * df["breadth_rank"].fillna(0.0)
        + THEME_SCORE_WEIGHTS["entropy_score"] * df["entropy_rank"].fillna(0.0)
        + THEME_SCORE_WEIGHTS["theme_rvol"] * df["rvol_rank"].fillna(0.0)
        + THEME_SCORE_WEIGHTS["rank_stability_5d"] * df["rank_stability_5d_rank"].fillna(0.5)
    ) / weight_total
    df["theme_heat_score"] = score
    df["theme_score"] = df["theme_heat_score"]
    tradable = df["is_tradable"].fillna(False).astype(bool)
    df["theme_heat_rank"] = np.nan
    df.loc[tradable, "theme_heat_rank"] = df.loc[tradable].groupby("date")["theme_heat_score"].rank(ascending=False, method="first")
    df["theme_rank"] = df["theme_heat_rank"]
    df["watchlist_rank"] = np.nan
    df.loc[~tradable, "watchlist_rank"] = df.loc[~tradable].groupby("date")["theme_heat_score"].rank(ascending=False, method="first")

    df["regime_rank_20d_raw"] = dense_rank_by_date(df, "theme_vs_spy_20d", ascending=False)
    df["rank_stability_10d"] = rolling_rank_stability(df["regime_rank_20d_raw"], df["date"], df["theme"], 10)
    df["rank_stability_20d"] = rolling_rank_stability(df["regime_rank_20d_raw"], df["date"], df["theme"], 20)
    regime_components = {
        "theme_vs_spy_10d": pct_rank_by_date(df, "theme_vs_spy_10d", ascending=True).fillna(0.0),
        "theme_vs_spy_20d": pct_rank_by_date(df, "theme_vs_spy_20d", ascending=True).fillna(0.0),
        "theme_breadth_10d": pct_rank_by_date(df, "theme_breadth_10d", ascending=True).fillna(0.0),
        "theme_breadth_20d": pct_rank_by_date(df, "theme_breadth_20d", ascending=True).fillna(0.0),
        "theme_above_20d_pct": pct_rank_by_date(df, "theme_above_20d_pct", ascending=True).fillna(0.0),
        "theme_rvol_10d": pct_rank_by_date(df, "theme_rvol_10d", ascending=True).fillna(0.0),
        "rank_stability_20d": pct_rank_by_date(df, "rank_stability_20d", ascending=True).fillna(0.5),
    }
    df["theme_regime_score"] = (
        0.20 * regime_components["theme_vs_spy_10d"]
        + 0.20 * regime_components["theme_vs_spy_20d"]
        + 0.15 * regime_components["theme_breadth_10d"]
        + 0.15 * regime_components["theme_breadth_20d"]
        + 0.10 * regime_components["theme_above_20d_pct"]
        + 0.10 * regime_components["theme_rvol_10d"]
        + 0.10 * regime_components["rank_stability_20d"]
    )
    df["theme_regime_rank"] = np.nan
    df.loc[tradable, "theme_regime_rank"] = df.loc[tradable].groupby("date")["theme_regime_score"].rank(
        ascending=False,
        method="first",
    )
    df = df.sort_values(["theme", "date"])
    df["theme_rank_change_1d"] = df.groupby("theme")["theme_rank"].diff(1) * -1.0
    df["theme_rank_change_5d"] = df.groupby("theme")["theme_rank"].diff(5) * -1.0
    by_date = df.groupby("date", group_keys=False)
    current_rank_score = 1.0 - (df["theme_rank"] - 1.0) / by_date["theme_rank"].transform("max").sub(1.0).replace(0, np.nan)
    rank_change_score = by_date["theme_rank_change_5d"].rank(pct=True, ascending=True).fillna(0.5)
    entropy_rank = by_date["entropy_score"].rank(pct=True, ascending=True).fillna(0.0)
    df["theme_persistence_score"] = (
        0.4 * current_rank_score.fillna(0.0)
        + 0.3 * rank_change_score
        + 0.2 * df["theme_breadth"].fillna(0.0)
        + 0.1 * entropy_rank
    )

    keep = [
        "date",
        "theme",
        "is_tradable",
        "is_watchlist_only",
        "theme_heat_score",
        "theme_heat_rank",
        "theme_regime_score",
        "theme_regime_rank",
        "theme_score",
        "theme_rank",
        "watchlist_rank",
        "theme_rank_change_1d",
        "theme_rank_change_5d",
        "rank_stability_5d",
        "theme_persistence_score",
        "theme_return_1d",
        "theme_return_3d",
        "theme_return_5d",
        "theme_return_10d",
        "theme_return_20d",
        "theme_vs_spy_5d",
        "theme_vs_spy_10d",
        "theme_vs_spy_20d",
        "theme_vs_qqq_5d",
        "theme_vs_qqq_10d",
        "theme_vs_qqq_20d",
        "theme_breadth",
        "theme_breadth_10d",
        "theme_breadth_20d",
        "theme_rvol",
        "theme_rvol_10d",
        "theme_advancers_pct",
        "theme_above_20d_pct",
        "theme_above_50d_pct",
        "theme_new_high_pct",
        "theme_dollar_volume",
        "theme_concentration_top3",
        "leader_concentration",
        "entropy_score",
        "theme_effective_breadth",
        "theme_num_constituents",
        "theme_effective_constituents",
        "rank_stability_10d",
        "rank_stability_20d",
        "leader_gap",
    ]
    out = df[keep].replace([np.inf, -np.inf], np.nan).sort_values(["date", "theme_rank"])
    out.to_parquet(THEME_SCORES_PATH, index=False)
    print(f"saved {len(out):,} theme score rows -> {THEME_SCORES_PATH}")


if __name__ == "__main__":
    main()
