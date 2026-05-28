from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    DAILY_BARS_PATH,
    LIVE_THEME_SIGNAL_RANKING_PATH,
    REPORT_DIR,
    THEME_DAILY_PATH,
    THEME_SCORES_PATH,
    THEME_SIGNAL_LABELS_PATH,
    ensure_dirs,
    load_theme_definitions,
)


SIGNAL_LABELS_PATH = THEME_SIGNAL_LABELS_PATH
LIVE_SIGNAL_RANKING_PATH = LIVE_THEME_SIGNAL_RANKING_PATH
SUMMARY_PATH = REPORT_DIR / "theme_signal_label_summary.csv"
PLAYBOOK_SUMMARY_PATH = REPORT_DIR / "theme_playbook_forward_summary.csv"
DICTIONARY_PATH = REPORT_DIR / "theme_signal_label_dictionary.csv"
JSON_SUMMARY_PATH = REPORT_DIR / "theme_signal_label_summary.json"

LABEL_HORIZONS = (5, 10, 20)
RETURN_HORIZONS = (1, 5, 10, 20)
EXTREME_HORIZONS = (5, 10, 20)
SHORT_ABS_THRESHOLDS = {5: -0.025, 10: -0.040, 20: -0.060}
SHORT_EXCESS_THRESHOLDS = {5: -0.015, 10: -0.025, 20: -0.040}
DD_RISK_THRESHOLDS = {5: -0.040, 10: -0.060, 20: -0.090}
ETF_EXCLUSIONS = {"SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "SMH", "XLK", "XLF", "XLV", "XLE"}


def clip01(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(0.0, 1.0)


def pct_rank_by_date(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df.groupby("date")[column].rank(pct=True, ascending=ascending)


def decile_from_rank(rank: pd.Series, count: pd.Series) -> pd.Series:
    decile = np.ceil(rank.astype(float) / count.astype(float) * 10.0)
    return decile.clip(1, 10)


def label_from_condition(valid: pd.Series, condition: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=condition.index, dtype="float64")
    out.loc[valid] = condition.loc[valid].astype(float)
    return out


def load_market_features() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    wide = bars.pivot(index="date", columns="ticker", values="px").sort_index()

    defs = load_theme_definitions()
    benchmark_tickers = {"SPY", "QQQ"}
    if not defs.empty and "benchmark" in defs.columns:
        benchmark_tickers |= set(defs["benchmark"].dropna().astype(str).str.upper())

    feature_blocks: list[pd.DataFrame] = []
    for ticker in sorted(t for t in benchmark_tickers if t in wide.columns):
        if ticker not in wide.columns:
            continue
        lower = ticker.lower()
        px = wide[ticker]
        block = pd.DataFrame(index=wide.index)
        block[f"{lower}_return_5d"] = px.pct_change(5)
        block[f"{lower}_return_10d"] = px.pct_change(10)
        block[f"{lower}_return_20d"] = px.pct_change(20)
        block[f"{lower}_return_63d"] = px.pct_change(63)
        block[f"{lower}_above_50dma"] = px > px.rolling(50, min_periods=30).mean()
        block[f"{lower}_above_200dma"] = px > px.rolling(200, min_periods=100).mean()
        block[f"{lower}_drawdown_63d"] = px / px.rolling(63, min_periods=20).max() - 1.0
        block[f"{lower}_drawdown_126d"] = px / px.rolling(126, min_periods=40).max() - 1.0
        for horizon in RETURN_HORIZONS:
            block[f"{lower}_fwd_return_{horizon}d"] = px.shift(-horizon) / px - 1.0
        feature_blocks.append(block)

    out = pd.concat(feature_blocks, axis=1).copy() if feature_blocks else pd.DataFrame(index=wide.index)

    stock_cols = [col for col in wide.columns if col not in ETF_EXCLUSIONS]
    stock_px = wide[stock_cols]
    out["market_breadth_50dma"] = (stock_px > stock_px.rolling(50, min_periods=30).mean()).mean(axis=1)
    out["market_breadth_200dma"] = (stock_px > stock_px.rolling(200, min_periods=100).mean()).mean(axis=1)
    out["market_breadth_200dma_delta_10d"] = out["market_breadth_200dma"] - out["market_breadth_200dma"].shift(10)
    out["market_breadth_50dma_delta_10d"] = out["market_breadth_50dma"] - out["market_breadth_50dma"].shift(10)

    out["signal_market_recovery_impulse"] = (
        (out["spy_drawdown_63d"].shift(5) < -0.08)
        & (out["spy_return_5d"] > 0.025)
        & (out["qqq_return_5d"] > 0.025)
        & (out["market_breadth_200dma_delta_10d"] > 0.025)
    )
    out["signal_market_risk_off"] = (
        ((~out["spy_above_200dma"].fillna(False)) & (~out["qqq_above_200dma"].fillna(False)))
        | (out["market_breadth_200dma"] < 0.40)
        | ((out["spy_drawdown_63d"] < -0.12) & (out["spy_return_20d"] < -0.05))
    )
    out["signal_market_risk_on"] = (
        out["spy_above_200dma"].fillna(False)
        & out["qqq_above_200dma"].fillna(False)
        & (out["market_breadth_200dma"] > 0.55)
        & (out["spy_return_20d"] > -0.01)
    )
    out["signal_market_playbook"] = np.select(
        [
            out["signal_market_recovery_impulse"].fillna(False),
            out["signal_market_risk_off"].fillna(False),
            out["signal_market_risk_on"].fillna(False),
        ],
        ["recovery_rotation", "risk_off_defense", "risk_on_continuation"],
        default="neutral_chop",
    )
    return out.reset_index().rename(columns={"index": "date"})


def add_future_extremes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").copy()
    px = frame["theme_index"].astype(float)
    for horizon in EXTREME_HORIZONS:
        future_paths = pd.concat([(px.shift(-step) / px - 1.0) for step in range(1, horizon + 1)], axis=1)
        frame[f"fwd_theme_min_return_{horizon}d"] = future_paths.min(axis=1)
        frame[f"fwd_theme_max_return_{horizon}d"] = future_paths.max(axis=1)
    return frame


def add_benchmark_forward_returns(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    defs = load_theme_definitions()
    meta_cols = ["theme", "category", "benchmark", "is_tradable", "is_watchlist_only"]
    meta = defs[[col for col in meta_cols if col in defs.columns]].copy()
    if not meta.empty:
        meta["theme"] = meta["theme"].astype(str).str.lower()
        df = df.drop(columns=[c for c in ["category", "benchmark"] if c in df.columns], errors="ignore")
        df = df.merge(meta[["theme", "category", "benchmark"]], on="theme", how="left")
    df["category"] = df["category"].fillna("Unknown")
    df["benchmark"] = df["benchmark"].fillna("SPY").astype(str).str.upper()

    market_cols = [c for c in market.columns if c.endswith(tuple(f"_fwd_return_{h}d" for h in RETURN_HORIZONS))]
    missing_market_cols = [col for col in market_cols if col not in df.columns]
    if missing_market_cols:
        market_fwd = market[["date", *missing_market_cols]].copy()
        df = df.merge(market_fwd, on="date", how="left")
    df = df.copy()
    available_benchmarks = {
        col.split("_fwd_return_")[0].upper()
        for col in market_cols
        if "_fwd_return_" in col
    }
    df["benchmark_available"] = df["benchmark"].isin(available_benchmarks)
    df.loc[~df["benchmark_available"], "benchmark"] = "SPY"

    for horizon in RETURN_HORIZONS:
        bench_col = f"benchmark_fwd_return_{horizon}d"
        df[bench_col] = df.get(f"spy_fwd_return_{horizon}d", np.nan)
        for benchmark in sorted(df["benchmark"].dropna().unique()):
            col = f"{benchmark.lower()}_fwd_return_{horizon}d"
            if col in df.columns:
                df.loc[df["benchmark"].eq(benchmark), bench_col] = df.loc[df["benchmark"].eq(benchmark), col]
    return df


def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["theme", "date"]).copy()
    grouped = df.groupby("theme")
    df["signal_regime_rank_delta_5d"] = grouped["theme_regime_rank"].diff(5) * -1.0
    df["signal_regime_rank_delta_20d"] = grouped["theme_regime_rank"].diff(20) * -1.0
    df["signal_rank_decay_5d"] = df["signal_regime_rank_delta_5d"] * -1.0
    df["signal_rank_decay_20d"] = df["signal_regime_rank_delta_20d"] * -1.0
    df["signal_breadth_delta_5d"] = grouped["theme_above_20d_pct"].diff(5)
    df["signal_breadth_delta_20d"] = grouped["theme_above_20d_pct"].diff(20)
    df["signal_relative_return_accel_5v20"] = df["theme_vs_spy_5d"] - (df["theme_vs_spy_20d"] / 4.0)
    df["signal_absolute_return_accel_5v20"] = df["theme_return_5d"] - (df["theme_return_20d"] / 4.0)

    tradable_count = df.groupby("date")["theme_regime_rank"].transform("count")
    df["signal_regime_rank_decile"] = decile_from_rank(df["theme_regime_rank"], tradable_count)
    df["signal_heat_rank_decile"] = decile_from_rank(df["theme_heat_rank"], tradable_count)
    df["signal_rank_percentile"] = 1.0 - (df["theme_regime_rank"] - 1.0) / (tradable_count - 1.0).replace(0, np.nan)
    df["signal_rank_percentile"] = clip01(df["signal_rank_percentile"].fillna(0.0))
    df["signal_weak_rank_score"] = 1.0 - df["signal_rank_percentile"]

    df["signal_relative_strength_score"] = pct_rank_by_date(df, "theme_vs_spy_20d", ascending=True).fillna(0.0)
    df["signal_relative_weakness_score"] = 1.0 - df["signal_relative_strength_score"]
    df["signal_absolute_trend_score"] = pct_rank_by_date(df, "theme_return_20d", ascending=True).fillna(0.0)
    df["signal_absolute_weakness_score"] = 1.0 - df["signal_absolute_trend_score"]
    df["signal_rank_improvement_score"] = pct_rank_by_date(df, "signal_regime_rank_delta_5d", ascending=True).fillna(0.5)
    df["signal_rank_decay_score"] = pct_rank_by_date(df, "signal_rank_decay_5d", ascending=True).fillna(0.5)
    df["signal_breadth_score"] = pct_rank_by_date(df, "theme_above_20d_pct", ascending=True).fillna(0.0)
    df["signal_low_breadth_score"] = 1.0 - df["signal_breadth_score"]
    df["signal_breadth_improvement_score"] = pct_rank_by_date(df, "signal_breadth_delta_5d", ascending=True).fillna(0.5)
    df["signal_breadth_decay_score"] = pct_rank_by_date(df, "signal_breadth_delta_5d", ascending=False).fillna(0.5)
    df["signal_rvol_score"] = pct_rank_by_date(df, "theme_rvol_10d", ascending=True).fillna(0.0)
    df["signal_recent_bounce_score"] = pct_rank_by_date(df, "theme_return_5d", ascending=True).fillna(0.0)
    df["signal_slowing_score"] = pct_rank_by_date(df, "signal_relative_return_accel_5v20", ascending=False).fillna(0.5)
    df["signal_leader_concentration_score"] = pct_rank_by_date(df, "theme_concentration_top3", ascending=True).fillna(0.0)

    df["signal_theme_decay_score"] = clip01(
        0.30 * df["signal_rank_decay_score"]
        + 0.25 * df["signal_relative_weakness_score"]
        + 0.20 * df["signal_breadth_decay_score"]
        + 0.15 * df["signal_absolute_weakness_score"]
        + 0.10 * df["signal_low_breadth_score"]
    )
    df["signal_theme_exhaustion_score"] = clip01(
        0.25 * df["signal_relative_strength_score"]
        + 0.25 * df["signal_slowing_score"]
        + 0.20 * df["signal_breadth_decay_score"]
        + 0.15 * df["signal_rank_decay_score"]
        + 0.15 * df["signal_leader_concentration_score"]
    )
    df["signal_theme_short_score"] = clip01(
        0.25 * df["signal_weak_rank_score"]
        + 0.20 * df["signal_relative_weakness_score"]
        + 0.20 * df["signal_absolute_weakness_score"]
        + 0.15 * df["signal_low_breadth_score"]
        + 0.10 * df["signal_theme_decay_score"]
        + 0.10 * (1.0 - df["signal_rank_improvement_score"])
    )
    df["signal_theme_hedge_score"] = clip01(
        0.35 * df["signal_weak_rank_score"]
        + 0.25 * df["signal_relative_weakness_score"]
        + 0.15 * df["signal_low_breadth_score"]
        + 0.15 * df["signal_theme_decay_score"]
        + 0.10 * (1.0 - pct_rank_by_date(df, "theme_vs_spy_10d", ascending=True).fillna(0.5))
    )
    df["signal_theme_recovery_score"] = clip01(
        0.25 * df["signal_weak_rank_score"]
        + 0.25 * df["signal_rank_improvement_score"]
        + 0.20 * df["signal_breadth_improvement_score"]
        + 0.15 * df["signal_recent_bounce_score"]
        + 0.15 * df["signal_market_recovery_impulse"].astype(float)
    )
    df["signal_theme_continuation_score"] = clip01(
        0.25 * df["signal_rank_percentile"]
        + 0.25 * df["signal_relative_strength_score"]
        + 0.20 * df["signal_breadth_score"]
        + 0.15 * df["theme_persistence_score"].fillna(0.0)
        + 0.10 * df["signal_rvol_score"]
        + 0.05 * (1.0 - df["signal_theme_exhaustion_score"])
    )

    df["signal_exhaustion_flag"] = (
        (df["signal_theme_exhaustion_score"] >= 0.75)
        & (df["theme_vs_spy_20d"] > 0.0)
        & (df["signal_breadth_delta_5d"] < 0.0)
    )
    df["signal_short_candidate_flag"] = (
        (df["signal_theme_short_score"] >= 0.75)
        & (df["theme_return_20d"] < 0.0)
        & (df["theme_vs_spy_20d"] < 0.0)
        & (df["theme_above_20d_pct"] < 0.45)
        & ~df["signal_market_playbook"].eq("recovery_rotation")
    )
    df["signal_short_flag"] = (
        df["signal_short_candidate_flag"]
        & (df["signal_theme_short_score"] >= 0.82)
        & df["signal_market_playbook"].eq("risk_off_defense")
    )
    df["signal_hedge_flag"] = (
        (
            ((df["signal_theme_hedge_score"] >= 0.70) & (df["theme_vs_spy_20d"] < 0.0))
            | df["signal_short_candidate_flag"]
        )
        & ~df["signal_market_playbook"].eq("recovery_rotation")
        & ~df["signal_short_flag"]
    )
    df["signal_recovery_long_flag"] = (
        (df["signal_theme_recovery_score"] >= 0.70)
        & df["signal_market_playbook"].eq("recovery_rotation")
        & (df["signal_regime_rank_decile"] >= 4)
        & ((df["theme_return_5d"] > 0.0) | (df["signal_regime_rank_delta_5d"] > 0.0))
    )
    df["signal_continuation_long_flag"] = (
        (df["signal_theme_continuation_score"] >= 0.72)
        & (df["signal_regime_rank_decile"] <= 2)
        & df["signal_market_playbook"].isin(["risk_on_continuation", "neutral_chop"])
        & ~df["signal_exhaustion_flag"]
    )
    df["signal_avoid_flag"] = (
        (df["signal_theme_hedge_score"] >= 0.62)
        & ~df["signal_short_flag"]
        & ~df["signal_recovery_long_flag"]
    )

    df["signal_pair_trade_side"] = np.select(
        [
            df["signal_short_flag"],
            df["signal_hedge_flag"],
            df["signal_recovery_long_flag"],
            df["signal_continuation_long_flag"],
            df["signal_avoid_flag"],
        ],
        ["short", "hedge_short", "long_recovery", "long_continuation", "avoid"],
        default="neutral",
    )
    return df


def add_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["theme", "date"]).copy()
    grouped = df.groupby("theme")
    for horizon in RETURN_HORIZONS:
        df[f"fwd_theme_return_{horizon}d"] = grouped["theme_index"].shift(-horizon) / df["theme_index"] - 1.0
        df[f"fwd_theme_excess_spy_{horizon}d"] = df[f"fwd_theme_return_{horizon}d"] - df[f"spy_fwd_return_{horizon}d"]
        df[f"fwd_theme_excess_qqq_{horizon}d"] = df[f"fwd_theme_return_{horizon}d"] - df[f"qqq_fwd_return_{horizon}d"]
        df[f"fwd_theme_excess_benchmark_{horizon}d"] = (
            df[f"fwd_theme_return_{horizon}d"] - df[f"benchmark_fwd_return_{horizon}d"]
        )
        df[f"future_regime_rank_{horizon}d"] = grouped["theme_regime_rank"].shift(-horizon)
        df[f"future_rank_improvement_{horizon}d"] = df["theme_regime_rank"] - df[f"future_regime_rank_{horizon}d"]

    df = pd.concat(
        [add_future_extremes(frame).assign(theme=theme) for theme, frame in df.groupby("theme", sort=False)],
        ignore_index=True,
    )

    tradable_count = df.groupby("date")["theme_regime_rank"].transform("count")
    for horizon in LABEL_HORIZONS:
        valid = df[f"fwd_theme_return_{horizon}d"].notna()
        fwd_rank = df.groupby("date")[f"fwd_theme_excess_benchmark_{horizon}d"].rank(ascending=False, method="first")
        fwd_decile = decile_from_rank(fwd_rank, tradable_count)
        future_rank_decile = decile_from_rank(df[f"future_regime_rank_{horizon}d"], tradable_count)
        rank_improve_decile = decile_from_rank(
            df.groupby("date")[f"future_rank_improvement_{horizon}d"].rank(ascending=False, method="first"),
            tradable_count,
        )

        df[f"label_fwd_excess_decile_{horizon}d"] = fwd_decile
        df[f"label_future_rank_decile_{horizon}d"] = future_rank_decile
        df[f"label_rank_improvement_decile_{horizon}d"] = rank_improve_decile
        df[f"label_forward_top_decile_{horizon}d"] = label_from_condition(valid, fwd_decile.eq(1))
        df[f"label_forward_bottom_decile_{horizon}d"] = label_from_condition(valid, fwd_decile.eq(10))
        df[f"label_future_top5_rank_{horizon}d"] = label_from_condition(valid, df[f"future_regime_rank_{horizon}d"].le(5))
        df[f"label_future_top10_rank_{horizon}d"] = label_from_condition(valid, df[f"future_regime_rank_{horizon}d"].le(10))
        df[f"label_rank_improver_{horizon}d"] = label_from_condition(valid, rank_improve_decile.le(2))
        df[f"label_continuation_long_{horizon}d"] = label_from_condition(
            valid,
            (df["signal_regime_rank_decile"] <= 2)
            & (df[f"fwd_theme_excess_benchmark_{horizon}d"] > 0.0)
            & (fwd_decile <= 3),
        )
        df[f"label_recovery_rebound_{horizon}d"] = label_from_condition(
            valid,
            (df["signal_regime_rank_decile"] >= 5)
            & (df[f"fwd_theme_return_{horizon}d"] > 0.0)
            & (fwd_decile <= 2),
        )
        df[f"label_hedge_underperformer_{horizon}d"] = label_from_condition(
            valid,
            (fwd_decile >= 8) & (df[f"fwd_theme_excess_benchmark_{horizon}d"] <= SHORT_EXCESS_THRESHOLDS[horizon]),
        )
        df[f"label_true_short_{horizon}d"] = label_from_condition(
            valid,
            (df[f"fwd_theme_return_{horizon}d"] <= SHORT_ABS_THRESHOLDS[horizon])
            & (df[f"fwd_theme_excess_benchmark_{horizon}d"] <= SHORT_EXCESS_THRESHOLDS[horizon]),
        )
        df[f"label_drawdown_risk_{horizon}d"] = label_from_condition(
            df[f"fwd_theme_min_return_{horizon}d"].notna(),
            df[f"fwd_theme_min_return_{horizon}d"] <= DD_RISK_THRESHOLDS[horizon],
        )
        df[f"label_avoid_{horizon}d"] = label_from_condition(valid, fwd_decile >= 8)
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signals = [
        "signal_continuation_long_flag",
        "signal_recovery_long_flag",
        "signal_hedge_flag",
        "signal_short_flag",
        "signal_short_candidate_flag",
        "signal_avoid_flag",
        "signal_exhaustion_flag",
    ]
    for signal in signals:
        sub = df[df[signal].fillna(False)]
        for horizon in LABEL_HORIZONS:
            rows.append(
                {
                    "slice": signal,
                    "horizon": horizon,
                    "observations": int(sub[f"fwd_theme_return_{horizon}d"].notna().sum()),
                    "avg_fwd_return": sub[f"fwd_theme_return_{horizon}d"].mean(),
                    "avg_fwd_excess_benchmark": sub[f"fwd_theme_excess_benchmark_{horizon}d"].mean(),
                    "pct_negative": sub[f"fwd_theme_return_{horizon}d"].lt(0).mean(),
                    "label_true_short_rate": sub[f"label_true_short_{horizon}d"].mean(),
                    "label_top_decile_rate": sub[f"label_forward_top_decile_{horizon}d"].mean(),
                }
            )
    return pd.DataFrame(rows)


def build_playbook_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for playbook, frame in df.groupby("signal_market_playbook"):
        for bucket, sub in frame.groupby("signal_regime_rank_decile"):
            for horizon in LABEL_HORIZONS:
                rows.append(
                    {
                        "playbook": playbook,
                        "rank_decile": int(bucket) if pd.notna(bucket) else np.nan,
                        "horizon": horizon,
                        "observations": int(sub[f"fwd_theme_return_{horizon}d"].notna().sum()),
                        "avg_fwd_return": sub[f"fwd_theme_return_{horizon}d"].mean(),
                        "avg_fwd_excess_benchmark": sub[f"fwd_theme_excess_benchmark_{horizon}d"].mean(),
                        "pct_negative": sub[f"fwd_theme_return_{horizon}d"].lt(0).mean(),
                        "top_decile_rate": sub[f"label_forward_top_decile_{horizon}d"].mean(),
                        "true_short_rate": sub[f"label_true_short_{horizon}d"].mean(),
                    }
                )
    return pd.DataFrame(rows).sort_values(["playbook", "horizon", "rank_decile"])


def build_latest_ranking(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df["date"].max()
    latest = df[df["date"].eq(latest_date)].copy()
    latest["signal_long_score"] = latest[["signal_theme_continuation_score", "signal_theme_recovery_score"]].max(axis=1)
    latest["signal_short_or_hedge_score"] = latest[["signal_theme_short_score", "signal_theme_hedge_score"]].max(axis=1)
    latest["signal_priority_score"] = latest[["signal_long_score", "signal_short_or_hedge_score"]].max(axis=1)
    keep = [
        "date",
        "theme",
        "category",
        "benchmark",
        "signal_market_playbook",
        "signal_pair_trade_side",
        "theme_regime_rank",
        "signal_regime_rank_decile",
        "theme_heat_rank",
        "theme_return_5d",
        "theme_return_20d",
        "theme_vs_spy_20d",
        "theme_above_20d_pct",
        "signal_regime_rank_delta_5d",
        "signal_breadth_delta_5d",
        "signal_theme_continuation_score",
        "signal_theme_recovery_score",
        "signal_theme_hedge_score",
        "signal_theme_short_score",
        "signal_theme_decay_score",
        "signal_theme_exhaustion_score",
        "signal_continuation_long_flag",
        "signal_recovery_long_flag",
        "signal_hedge_flag",
        "signal_short_flag",
        "signal_short_candidate_flag",
        "signal_avoid_flag",
        "signal_priority_score",
    ]
    return latest[keep].sort_values(["signal_pair_trade_side", "signal_priority_score"], ascending=[True, False])


def build_dictionary() -> pd.DataFrame:
    rows = [
        ("signal_market_playbook", "feature", "Rule-based higher-timeframe regime: risk-on continuation, recovery rotation, risk-off defense, or neutral chop."),
        ("signal_theme_continuation_score", "feature", "Live score for already-strong themes that can keep leading."),
        ("signal_theme_recovery_score", "feature", "Live score for weak or mid-ranked themes showing rebound behavior during recovery regimes."),
        ("signal_theme_hedge_score", "feature", "Live score for weak themes that are better avoid/hedge candidates than outright shorts."),
        ("signal_theme_short_score", "feature", "Live score for true downside continuation candidates, requiring weak rank, weak trend, poor breadth, and no recovery playbook."),
        ("signal_short_candidate_flag", "feature", "Weak absolute/relative breakdown setup; treated as hedge-short in risk-on and true short only in risk-off defense."),
        ("signal_theme_decay_score", "feature", "Live deterioration score from rank decay, relative weakness, breadth decay, and absolute weakness."),
        ("signal_theme_exhaustion_score", "feature", "Live score for crowded/extended leaders showing slowing momentum or breadth decay."),
        ("signal_pair_trade_side", "feature", "Single live role for portfolio construction: long continuation, long recovery, hedge short, short, avoid, or neutral."),
        ("label_forward_top_decile_*d", "label", "Forward benchmark-excess return finished in the top decile for the horizon."),
        ("label_future_top5_rank_*d", "label", "Theme became or remained a top-5 regime-rank theme at the horizon."),
        ("label_rank_improver_*d", "label", "Theme was a top rank improver by the horizon."),
        ("label_continuation_long_*d", "label", "Current strong theme produced positive benchmark-excess return and top-third forward outcome."),
        ("label_recovery_rebound_*d", "label", "Current mid/weak-ranked theme produced positive return and top-quintile forward outcome."),
        ("label_hedge_underperformer_*d", "label", "Theme underperformed its benchmark enough to be useful as a hedge leg."),
        ("label_true_short_*d", "label", "Theme had negative absolute forward return and negative benchmark-excess return beyond threshold."),
        ("label_drawdown_risk_*d", "label", "Theme experienced a forward adverse move beyond the horizon drawdown threshold."),
        ("label_avoid_*d", "label", "Theme landed in weak forward benchmark-excess deciles, useful for avoid/underweight training."),
    ]
    return pd.DataFrame(rows, columns=["name", "kind", "description"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal theme signals and forward ML labels.")
    parser.add_argument("--start", default=None, help="Optional YYYY-MM-DD start date for output rows.")
    args = parser.parse_args()

    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(THEME_SCORES_PATH)
    daily = pd.read_parquet(THEME_DAILY_PATH)[["date", "theme", "theme_index"]]
    market = load_market_features()
    for frame in (scores, daily, market):
        frame["date"] = pd.to_datetime(frame["date"])

    df = scores.merge(daily, on=["date", "theme"], how="left")
    df = df.merge(market, on="date", how="left")
    df = add_benchmark_forward_returns(df, market)
    df = add_signal_features(df)
    df = add_forward_labels(df)
    if args.start:
        df = df[df["date"] >= pd.Timestamp(args.start)].copy()

    signal_cols = [col for col in df.columns if col.startswith("signal_")]
    label_cols = [col for col in df.columns if col.startswith("label_")]
    output_cols = [
        "date",
        "theme",
        "category",
        "benchmark",
        "is_tradable",
        "is_watchlist_only",
        "theme_regime_rank",
        "theme_heat_rank",
        "theme_return_5d",
        "theme_return_10d",
        "theme_return_20d",
        "theme_vs_spy_5d",
        "theme_vs_spy_10d",
        "theme_vs_spy_20d",
        "theme_above_20d_pct",
        "theme_above_50d_pct",
        "theme_breadth",
        "theme_rvol",
        "theme_persistence_score",
        "theme_index",
        *signal_cols,
        *[col for col in df.columns if col.startswith("fwd_") or col.startswith("future_")],
        *label_cols,
    ]
    output_cols = [col for col in dict.fromkeys(output_cols) if col in df.columns]
    out = df[output_cols].replace([np.inf, -np.inf], np.nan).sort_values(["date", "theme"])
    out.to_parquet(SIGNAL_LABELS_PATH, index=False)

    latest = build_latest_ranking(out)
    latest.to_csv(LIVE_SIGNAL_RANKING_PATH, index=False)
    summary = build_summary(out)
    summary.to_csv(SUMMARY_PATH, index=False)
    playbook = build_playbook_summary(out)
    playbook.to_csv(PLAYBOOK_SUMMARY_PATH, index=False)
    dictionary = build_dictionary()
    dictionary.to_csv(DICTIONARY_PATH, index=False)

    latest_counts = latest["signal_pair_trade_side"].value_counts().to_dict()
    payload = {
        "rows": int(len(out)),
        "themes": int(out["theme"].nunique()),
        "start_date": str(out["date"].min().date()),
        "end_date": str(out["date"].max().date()),
        "latest_signal_counts": {str(k): int(v) for k, v in latest_counts.items()},
        "outputs": {
            "signal_labels": str(SIGNAL_LABELS_PATH),
            "live_ranking": str(LIVE_SIGNAL_RANKING_PATH),
            "summary": str(SUMMARY_PATH),
            "playbook_summary": str(PLAYBOOK_SUMMARY_PATH),
            "dictionary": str(DICTIONARY_PATH),
        },
    }
    JSON_SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"saved {len(out):,} signal/label rows -> {SIGNAL_LABELS_PATH}")
    print(f"saved live ranking -> {LIVE_SIGNAL_RANKING_PATH}")
    print(f"latest signal counts: {latest_counts}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
