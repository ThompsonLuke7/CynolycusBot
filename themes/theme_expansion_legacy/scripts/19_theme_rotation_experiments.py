from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    BACKTEST_DIR,
    HOLD_DAYS,
    LEADERS_PER_THEME,
    OUTPUT_DIR,
    THEME_SCORES_PATH,
    TOP_N_THEMES,
    TRANSACTION_COST_BPS,
    ensure_dirs,
)


RESULTS_PATH = OUTPUT_DIR / "theme_rotation_experiments.csv"


def load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_script("07_backtest_ranked_themes.py", "baseline_backtest")
leader_builder = load_script("05_rank_theme_leaders.py", "leader_builder")


def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(THEME_SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    if "is_tradable" in scores.columns:
        scores = scores[scores["is_tradable"].fillna(False).astype(bool)].copy()
    return scores


def build_all_member_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    panel = leader_builder.load_stock_theme_panel()
    panel = panel.merge(scores[["date", "theme", "theme_return_5d"]], on=["date", "theme"], how="inner")
    panel["stock_vs_theme_5d"] = panel["stock_return_5d"] - panel["theme_return_5d"]
    panel = leader_builder.add_leader_follower_scores(panel)
    panel["leader_rank"] = panel["ticker_rank_in_theme"]
    return panel.replace([np.inf, -np.inf], np.nan)


def leaders_from_ranks(panel: pd.DataFrame, min_rank: int, max_rank: int) -> pd.DataFrame:
    out = panel[(panel["leader_rank"] >= min_rank) & (panel["leader_rank"] <= max_rank)].copy()
    return out[
        [
            "date",
            "theme",
            "ticker",
            "leader_score",
            "leader_rank",
            "stock_return_1d",
            "stock_return_5d",
            "stock_vs_theme_5d",
            "ticker_lag_vs_leader",
        ]
    ]


def oracle_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy().sort_values(["theme", "date"])
    out["oracle_next_5d_theme_return"] = out.groupby("theme")["theme_return_5d"].shift(-HOLD_DAYS)
    out["theme_regime_rank"] = out.groupby("date")["oracle_next_5d_theme_return"].rank(ascending=False, method="first")
    return out


def stability_gate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    daily_cutoff = out.groupby("date")["rank_stability_5d"].transform("median")
    eligible = out["rank_stability_5d"] >= daily_cutoff
    out.loc[~eligible, "theme_regime_rank"] = 999.0
    return out


def persistent_top5_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy().sort_values(["theme", "date"])
    out["top5_recent_count"] = (
        out["theme_regime_rank"]
        .le(5)
        .groupby(out["theme"])
        .transform(lambda s: s.rolling(5, min_periods=3).sum())
    )
    out.loc[out["top5_recent_count"] < 2, "theme_regime_rank"] = 999.0
    return out


def run_long_only(
    label: str,
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    top_themes: int = TOP_N_THEMES,
    leaders_per_theme: int = LEADERS_PER_THEME,
) -> dict[str, float | str]:
    bt = baseline.run_backtest(scores, leaders, returns, top_themes, leaders_per_theme)
    stats = baseline.perf_stats(bt["strategy_return"], bt["equity"], bt["turnover"])
    return {
        "experiment": label,
        "cagr": stats["cagr"],
        "sharpe": stats["sharpe"],
        "sortino": stats["sortino"],
        "max_drawdown": stats["max_drawdown"],
        "profit_factor": stats["profit_factor"],
        "turnover": stats["turnover"],
        "hit_rate": stats["hit_rate"],
    }


def select_daily_weights(
    date: pd.Timestamp,
    scores: pd.DataFrame,
    panel_by_date: dict[pd.Timestamp, pd.DataFrame],
    regime: str,
    short_ratio: float,
) -> pd.Series:
    day_scores = scores[scores["date"].eq(date)].dropna(subset=["theme_regime_rank"])
    if day_scores.empty or date not in panel_by_date:
        return pd.Series(dtype=float)

    exposure = baseline.exposure_for(regime)
    if exposure <= 0:
        return pd.Series(dtype=float)

    longs = day_scores.nsmallest(TOP_N_THEMES, "theme_regime_rank")["theme"].tolist()
    shorts = day_scores.nlargest(TOP_N_THEMES, "theme_regime_rank")["theme"].tolist()
    day_panel = panel_by_date[date]

    long_rows = day_panel[day_panel["theme"].isin(longs) & day_panel["leader_rank"].between(1, LEADERS_PER_THEME)]
    short_pool = day_panel[day_panel["theme"].isin(shorts)].copy()
    short_rows = (
        short_pool.sort_values(["theme", "leader_rank"], ascending=[True, False])
        .groupby("theme", group_keys=False)
        .head(LEADERS_PER_THEME)
    )

    weights: dict[str, float] = {}
    if not long_rows.empty:
        long_tickers = sorted(long_rows["ticker"].dropna().unique())
        for ticker in long_tickers:
            weights[ticker] = weights.get(ticker, 0.0) + exposure / len(long_tickers)
    if short_ratio > 0 and not short_rows.empty:
        short_tickers = sorted(short_rows["ticker"].dropna().unique())
        for ticker in short_tickers:
            weights[ticker] = weights.get(ticker, 0.0) - (exposure * short_ratio) / len(short_tickers)

    return pd.Series(weights, dtype=float)


def run_long_short(
    label: str,
    scores: pd.DataFrame,
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    short_ratio: float,
) -> dict[str, float | str]:
    dates = baseline.build_trade_calendar(scores, returns)
    regimes = baseline.build_market_regime().set_index("date")["regime"]
    panel_by_date = {date: frame for date, frame in panel.groupby("date", sort=False)}

    prev_weights = pd.Series(dtype=float)
    rows = []
    for date in dates:
        if prev_weights.empty:
            gross_return = 0.0
        else:
            gross_return = float((returns.loc[date].reindex(prev_weights.index).fillna(0.0) * prev_weights).sum())

        regime = regimes.get(date, "neutral")
        next_weights = select_daily_weights(date, scores, panel_by_date, regime, short_ratio)
        turnover = float(next_weights.sub(prev_weights, fill_value=0.0).abs().sum())
        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append({"date": date, "strategy_return": gross_return - cost, "turnover": turnover})
        prev_weights = next_weights

    bt = pd.DataFrame(rows)
    bt["equity"] = (1.0 + bt["strategy_return"]).cumprod()
    stats = baseline.perf_stats(bt["strategy_return"], bt["equity"], bt["turnover"])
    return {
        "experiment": label,
        "cagr": stats["cagr"],
        "sharpe": stats["sharpe"],
        "sortino": stats["sortino"],
        "max_drawdown": stats["max_drawdown"],
        "profit_factor": stats["profit_factor"],
        "turnover": stats["turnover"],
        "hit_rate": stats["hit_rate"],
    }


def main() -> None:
    ensure_dirs()
    scores = load_scores()
    returns = baseline.load_stock_returns()
    panel = build_all_member_ranks(scores)
    current_leaders = leaders_from_ranks(panel, 1, 3)
    followers_4_6 = leaders_from_ranks(panel, 4, 6)
    followers_4_9 = leaders_from_ranks(panel, 4, 9)

    rows = [
        run_long_only("current_regime_leaders_1_3", scores, current_leaders, returns, leaders_per_theme=3),
        run_long_only("stability_gate_leaders_1_3", stability_gate_scores(scores), current_leaders, returns, leaders_per_theme=3),
        run_long_only("persistent_top5_2of5_leaders_1_3", persistent_top5_scores(scores), current_leaders, returns, leaders_per_theme=3),
        run_long_only("theme_followers_4_6", scores, followers_4_6, returns, leaders_per_theme=6),
        run_long_only("theme_followers_4_9", scores, followers_4_9, returns, leaders_per_theme=9),
        run_long_only("leaky_oracle_next5d_theme_current_leaders", oracle_scores(scores), current_leaders, returns, leaders_per_theme=3),
        run_long_short("long_leaders_short_worst_themes_half", scores, panel, returns, short_ratio=0.5),
        run_long_short("long_leaders_short_worst_themes_equal", scores, panel, returns, short_ratio=1.0),
    ]
    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out.to_csv(RESULTS_PATH, index=False)
    print(out.to_string(index=False))
    print(f"saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
