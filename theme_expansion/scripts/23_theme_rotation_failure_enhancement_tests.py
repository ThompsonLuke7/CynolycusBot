from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    BACKTEST_DIR,
    DAILY_BARS_PATH,
    OUTPUT_DIR,
    THEME_DAILY_PATH,
    THEME_SCORES_PATH,
    TRANSACTION_COST_BPS,
    ensure_dirs,
)


OUT_DIR = OUTPUT_DIR / "deep_theme_rotation"
ENHANCED_RESULTS_PATH = OUT_DIR / "enhanced_theme_index_results.csv"
ENHANCED_YEARLY_PATH = OUT_DIR / "enhanced_theme_index_yearly.csv"
ENHANCED_QUARTERLY_2025_PATH = OUT_DIR / "enhanced_theme_index_2025_quarterly.csv"
RANK_BUCKET_PERIOD_PATH = OUT_DIR / "rank_bucket_period_diagnostics.csv"
ACTIVE_THEME_ATTRIBUTION_PATH = OUT_DIR / "existing_active_theme_period_attribution.csv"
FAILURE_SUMMARY_PATH = OUT_DIR / "failure_enhancement_summary.json"

HOLD_DAYS = 5
TRADING_DAYS = 252.0


FilterFn = Callable[[pd.DataFrame], pd.Series]


def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(THEME_SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    if "is_tradable" in scores.columns:
        scores = scores[scores["is_tradable"].fillna(False).astype(bool)].copy()
    scores = scores.dropna(subset=["theme_regime_rank"]).sort_values(["theme", "date"]).copy()
    scores["fwd_5d_theme_return"] = scores.groupby("theme")["theme_return_5d"].shift(-HOLD_DAYS)
    scores["period_q"] = scores["date"].dt.to_period("Q").astype(str)
    scores["period_y"] = scores["date"].dt.to_period("Y").astype(str)

    for col, ascending in [
        ("theme_regime_rank", True),
        ("theme_heat_rank", True),
        ("theme_vs_spy_20d", False),
        ("theme_return_20d", False),
        ("theme_above_20d_pct", False),
        ("theme_rvol", False),
        ("rank_stability_5d", False),
    ]:
        rank_col = f"{col}_pct_rank"
        scores[rank_col] = scores.groupby("date")[col].rank(pct=True, ascending=ascending)

    scores["composite_trend_score"] = (
        0.45 * scores["theme_regime_rank_pct_rank"].fillna(0.5)
        + 0.20 * scores["theme_heat_rank_pct_rank"].fillna(0.5)
        + 0.20 * scores["theme_vs_spy_20d_pct_rank"].fillna(0.5)
        + 0.15 * scores["theme_above_20d_pct_pct_rank"].fillna(0.5)
    )
    scores["composite_trend_rank"] = scores.groupby("date")["composite_trend_score"].rank(method="first")
    scores["composite_resilience_score"] = (
        0.35 * scores["theme_regime_rank_pct_rank"].fillna(0.5)
        + 0.25 * scores["theme_above_20d_pct_pct_rank"].fillna(0.5)
        + 0.20 * scores["rank_stability_5d_pct_rank"].fillna(0.5)
        + 0.20 * scores["theme_rvol_pct_rank"].fillna(0.5)
    )
    scores["composite_resilience_rank"] = scores.groupby("date")["composite_resilience_score"].rank(method="first")
    daily_stability = scores.groupby("date")["rank_stability_5d"].transform("median")
    scores["filter_none"] = True
    scores["filter_abs_uptrend"] = (scores["theme_return_20d"] > 0) & (scores["theme_above_20d_pct"] >= 0.50)
    scores["filter_relative_uptrend"] = (scores["theme_vs_spy_20d"] > 0) & (scores["theme_return_20d"] > 0)
    scores["filter_breadth_confirm"] = (scores["theme_above_20d_pct"] >= 0.55) & (scores["theme_above_50d_pct"] >= 0.45)
    scores["filter_heat_confirm"] = scores["theme_heat_rank"] <= 20
    scores["filter_fresh_improving"] = (scores["theme_rank_change_5d"] <= 0) & (scores["rank_stability_5d"] >= daily_stability)
    scores["filter_no_breakdown"] = (scores["theme_return_5d"] > -0.03) & (scores["theme_above_20d_pct"] >= 0.35)
    scores["filter_trend_breadth_heat"] = (
        (scores["theme_return_20d"] > 0)
        & (scores["theme_vs_spy_20d"] > 0)
        & (scores["theme_above_20d_pct"] >= 0.50)
        & (scores["theme_heat_rank"] <= 35)
    )
    return scores.replace([np.inf, -np.inf], np.nan)


def load_theme_returns() -> pd.DataFrame:
    theme_daily = pd.read_parquet(THEME_DAILY_PATH)
    theme_daily["date"] = pd.to_datetime(theme_daily["date"])
    return theme_daily.pivot(index="date", columns="theme", values="theme_return_1d").sort_index()


def load_market_returns() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    bars = bars.sort_values(["ticker", "date"])
    bars["return_1d"] = bars.groupby("ticker")["px"].pct_change()
    returns = bars.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    return returns[[col for col in ["SPY", "QQQ", "IWM"] if col in returns.columns]].fillna(0.0)


def perf_stats(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    returns = returns.fillna(0.0)
    if turnover is None:
        turnover = pd.Series(0.0, index=returns.index)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    loss_sum = abs(losers.sum())
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 else np.nan
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    return {
        "total_return": final_equity - 1.0,
        "cagr": float(cagr) if pd.notna(cagr) else np.nan,
        "sharpe": float(np.sqrt(TRADING_DAYS) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if pd.notna(cagr) and max_drawdown < 0 else np.nan,
        "hit_rate": float((returns > 0).mean()),
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else np.nan,
        "turnover": float(turnover.fillna(0.0).mean()),
    }


def period_stats(daily: pd.DataFrame, label: str, freq: str) -> pd.DataFrame:
    frame = daily.copy()
    frame["period"] = frame["date"].dt.to_period(freq).astype(str)
    rows = []
    for period, group in frame.groupby("period", sort=True):
        stats = perf_stats(group["strategy_return"])
        rows.append({"period": period, "strategy": label, **stats, "days": int(len(group))})
    return pd.DataFrame(rows)


def market_features(market_returns: pd.DataFrame) -> pd.DataFrame:
    px = (1.0 + market_returns.fillna(0.0)).cumprod()
    out = pd.DataFrame(index=market_returns.index)
    out["spy_return_5d"] = px["SPY"].pct_change(5) if "SPY" in px else 0.0
    out["qqq_return_5d"] = px["QQQ"].pct_change(5) if "QQQ" in px else 0.0
    out["spy_return_20d"] = px["SPY"].pct_change(20) if "SPY" in px else 0.0
    out["spy_drawdown_63d"] = px["SPY"] / px["SPY"].rolling(63, min_periods=20).max() - 1.0 if "SPY" in px else 0.0
    out["qqq_above_20d"] = px["QQQ"] > px["QQQ"].rolling(20, min_periods=10).mean() if "QQQ" in px else False
    out["spy_recovery_burst"] = (
        (out["spy_return_5d"] > 0.035)
        & (out["qqq_return_5d"] > 0.040)
        & (out["spy_drawdown_63d"].shift(5) < -0.08)
    )
    out["risk_off"] = (out["spy_drawdown_63d"] < -0.12) & (out["spy_return_20d"] < -0.05)
    return out


def run_theme_strategy(
    *,
    label: str,
    rank_col: str,
    top_n: int,
    filter_name: str,
    scores: pd.DataFrame,
    theme_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    market: pd.DataFrame,
    scores_by_rank_date: dict[str, dict[pd.Timestamp, pd.DataFrame]],
    fallback: str,
    blend_spy: float = 0.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    dates = sorted(set(scores["date"]).intersection(theme_returns.index))
    prev_weights = pd.Series(dtype=float)
    rows: list[dict[str, object]] = []
    filter_col = f"filter_{filter_name}"

    for date in dates:
        gross_return = 0.0
        if not prev_weights.empty:
            theme_part = prev_weights[~prev_weights.index.isin(["SPY", "QQQ"])]
            index_part = prev_weights[prev_weights.index.isin(["SPY", "QQQ"])]
            if not theme_part.empty:
                gross_return += float((theme_returns.loc[date].reindex(theme_part.index).fillna(0.0) * theme_part).sum())
            if not index_part.empty:
                gross_return += float((market_returns.loc[date].reindex(index_part.index).fillna(0.0) * index_part).sum())

        day = scores_by_rank_date[rank_col][date]
        eligible = day[day[filter_col].fillna(False)].head(top_n)
        weights: dict[str, float] = {}
        if not eligible.empty:
            theme_weight_total = max(0.0, 1.0 - blend_spy)
            per_theme = theme_weight_total / len(eligible)
            for theme in eligible["theme"]:
                weights[str(theme)] = per_theme
            if blend_spy > 0 and "SPY" in market_returns.columns:
                weights["SPY"] = blend_spy

        feature = market.loc[date] if date in market.index else pd.Series(dtype=float)
        if fallback == "qqq_recovery" and bool(feature.get("spy_recovery_burst", False)):
            weights = {"QQQ": 1.0} if "QQQ" in market_returns.columns else {"SPY": 1.0}
        elif fallback == "cash_risk_off" and bool(feature.get("risk_off", False)):
            weights = {}
        elif fallback == "half_theme_half_qqq_recovery" and bool(feature.get("spy_recovery_burst", False)):
            if weights:
                weights = {key: value * 0.5 for key, value in weights.items()}
                weights["QQQ" if "QQQ" in market_returns.columns else "SPY"] = 0.5

        next_weights = pd.Series(weights, dtype=float)
        turnover = float(next_weights.sub(prev_weights, fill_value=0.0).abs().sum())
        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append(
            {
                "date": date,
                "strategy_return": gross_return - cost,
                "turnover": turnover,
                "selected_themes": "|".join([idx for idx in next_weights.index if idx not in {"SPY", "QQQ"}]),
                "fallback_active": fallback != "none" and ("SPY" in next_weights.index or "QQQ" in next_weights.index),
            }
        )
        prev_weights = next_weights

    daily = pd.DataFrame(rows)
    stats = perf_stats(daily["strategy_return"], daily["turnover"])
    return {
        "label": label,
        "rank_col": rank_col,
        "top_n": top_n,
        "filter": filter_name,
        "fallback": fallback,
        "blend_spy": blend_spy,
        **stats,
    }, daily


def build_enhancement_tests(
    scores: pd.DataFrame,
    theme_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    market = market_features(market_returns)
    rank_cols = ["theme_regime_rank", "composite_trend_rank", "composite_resilience_rank"]
    scores_by_rank_date = {
        rank_col: {date: day.sort_values(rank_col) for date, day in scores.groupby("date", sort=False)}
        for rank_col in rank_cols
    }
    configs: list[dict[str, object]] = []
    for rank_col in rank_cols:
        for top_n in [3, 5]:
            for filter_name in [
                "none",
                "abs_uptrend",
                "relative_uptrend",
                "breadth_confirm",
                "heat_confirm",
                "fresh_improving",
                "no_breakdown",
                "trend_breadth_heat",
            ]:
                configs.append(
                    {
                        "label": f"{rank_col}_top{top_n}_{filter_name}",
                        "rank_col": rank_col,
                        "top_n": top_n,
                        "filter_name": filter_name,
                        "fallback": "none",
                        "blend_spy": 0.0,
                    }
                )

    for blend_spy in [0.20, 0.35]:
        configs.append(
            {
                "label": f"theme_regime_rank_top5_no_breakdown_spyblend{int(blend_spy * 100)}",
                "rank_col": "theme_regime_rank",
                "top_n": 5,
                "filter_name": "no_breakdown",
                "fallback": "none",
                "blend_spy": blend_spy,
            }
        )
    for fallback in ["qqq_recovery", "half_theme_half_qqq_recovery", "cash_risk_off"]:
        configs.append(
            {
                "label": f"theme_regime_rank_top5_no_breakdown_{fallback}",
                "rank_col": "theme_regime_rank",
                "top_n": 5,
                "filter_name": "no_breakdown",
                "fallback": fallback,
                "blend_spy": 0.0,
            }
        )

    rows = []
    daily_by_label: dict[str, pd.DataFrame] = {}
    for cfg in configs:
        row, daily = run_theme_strategy(
            label=str(cfg["label"]),
            rank_col=str(cfg["rank_col"]),
            top_n=int(cfg["top_n"]),
            filter_name=str(cfg["filter_name"]),
            scores=scores,
            theme_returns=theme_returns,
            market_returns=market_returns,
            market=market,
            scores_by_rank_date=scores_by_rank_date,
            fallback=str(cfg["fallback"]),
            blend_spy=float(cfg["blend_spy"]),
        )
        rows.append(row)
        daily_by_label[row["label"]] = daily

    results = pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False)
    keep_labels = results.head(8)["label"].tolist() + [
        "theme_regime_rank_top5_none",
        "theme_regime_rank_top5_no_breakdown",
        "theme_regime_rank_top5_no_breakdown_qqq_recovery",
    ]
    yearly = []
    quarterly = []
    for label in dict.fromkeys(keep_labels):
        if label in daily_by_label:
            yearly.append(period_stats(daily_by_label[label], label, "Y"))
            quarterly.append(period_stats(daily_by_label[label], label, "Q"))
    yearly_out = pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame()
    quarterly_out = pd.concat(quarterly, ignore_index=True) if quarterly else pd.DataFrame()
    quarterly_out = quarterly_out[quarterly_out["period"].str.startswith("2025")].copy() if not quarterly_out.empty else quarterly_out
    return results, yearly_out, quarterly_out, daily_by_label


def build_rank_bucket_period_diagnostics(scores: pd.DataFrame, market_returns: pd.DataFrame) -> pd.DataFrame:
    spy_curve = (1.0 + market_returns["SPY"].fillna(0.0)).cumprod()
    spy_fwd = spy_curve.shift(-HOLD_DAYS) / spy_curve - 1.0
    data = scores.merge(spy_fwd.rename("fwd_5d_spy_return").reset_index(), on="date", how="left")
    data["fwd_5d_excess_vs_spy"] = data["fwd_5d_theme_return"] - data["fwd_5d_spy_return"]
    data["rank_bucket"] = pd.cut(
        data["theme_regime_rank"],
        bins=[0, 3, 5, 10, 25, 50, 75, 200],
        labels=["1-3", "4-5", "6-10", "11-25", "26-50", "51-75", "76+"],
        include_lowest=True,
    )
    periods = {
        "2025Q1": data["period_q"].eq("2025Q1"),
        "2025Q2": data["period_q"].eq("2025Q2"),
        "2025Q3": data["period_q"].eq("2025Q3"),
        "2025Q4": data["period_q"].eq("2025Q4"),
        "2026YTD": data["period_y"].eq("2026"),
        "full": pd.Series(True, index=data.index),
    }
    rows = []
    for period, mask in periods.items():
        sample = data[mask].copy()
        grouped = sample.groupby("rank_bucket", observed=True)
        out = grouped.agg(
            observations=("fwd_5d_theme_return", "count"),
            avg_fwd_5d_theme_return=("fwd_5d_theme_return", "mean"),
            median_fwd_5d_theme_return=("fwd_5d_theme_return", "median"),
            pct_negative_fwd_5d=("fwd_5d_theme_return", lambda s: float((s < 0).mean())),
            avg_fwd_5d_excess_vs_spy=("fwd_5d_excess_vs_spy", "mean"),
            avg_theme_return_20d=("theme_return_20d", "mean"),
            avg_above_20d_pct=("theme_above_20d_pct", "mean"),
        ).reset_index()
        out.insert(0, "period", period)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def build_existing_active_theme_attribution() -> pd.DataFrame:
    path = BACKTEST_DIR / "rule_based_theme_rotation_daily.parquet"
    if not path.exists():
        return pd.DataFrame()
    bt = pd.read_parquet(path)
    bt["date"] = pd.to_datetime(bt["date"])
    bt["period_q"] = bt["date"].dt.to_period("Q").astype(str)
    rows = []
    for row in bt.itertuples(index=False):
        themes = [theme for theme in str(row.active_themes).split("|") if theme and theme != "nan"]
        if not themes:
            continue
        per_theme_return = row.strategy_return / len(themes)
        for theme in themes:
            rows.append({"date": row.date, "period": row.period_q, "theme": theme, "rough_daily_contribution": per_theme_return})
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return expanded
    out = (
        expanded.groupby(["period", "theme"])
        .agg(
            active_days=("date", "nunique"),
            rough_total_contribution=("rough_daily_contribution", "sum"),
            avg_contribution_day=("rough_daily_contribution", "mean"),
        )
        .reset_index()
        .sort_values(["period", "rough_total_contribution"])
    )
    return out


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    theme_returns = load_theme_returns()
    market_returns = load_market_returns()

    enhanced, yearly, quarterly_2025, daily_by_label = build_enhancement_tests(scores, theme_returns, market_returns)
    enhanced.to_csv(ENHANCED_RESULTS_PATH, index=False)
    yearly.to_csv(ENHANCED_YEARLY_PATH, index=False)
    quarterly_2025.to_csv(ENHANCED_QUARTERLY_2025_PATH, index=False)

    rank_buckets = build_rank_bucket_period_diagnostics(scores, market_returns)
    rank_buckets.to_csv(RANK_BUCKET_PERIOD_PATH, index=False)

    attribution = build_existing_active_theme_attribution()
    attribution.to_csv(ACTIVE_THEME_ATTRIBUTION_PATH, index=False)

    q2_attribution = attribution[attribution["period"].eq("2025Q2")].head(15).to_dict(orient="records") if not attribution.empty else []
    q2_buckets = rank_buckets[rank_buckets["period"].eq("2025Q2")].to_dict(orient="records")
    y2026_buckets = rank_buckets[rank_buckets["period"].eq("2026YTD")].to_dict(orient="records")
    summary = {
        "top_enhanced_theme_index_configs": enhanced.head(20).to_dict(orient="records"),
        "enhanced_2025_quarterly": quarterly_2025.to_dict(orient="records"),
        "rank_bucket_2025q2": q2_buckets,
        "rank_bucket_2026ytd": y2026_buckets,
        "worst_existing_active_themes_2025q2": q2_attribution,
        "output_files": {
            "enhanced_results": str(ENHANCED_RESULTS_PATH),
            "enhanced_yearly": str(ENHANCED_YEARLY_PATH),
            "enhanced_2025_quarterly": str(ENHANCED_QUARTERLY_2025_PATH),
            "rank_bucket_periods": str(RANK_BUCKET_PERIOD_PATH),
            "active_theme_attribution": str(ACTIVE_THEME_ATTRIBUTION_PATH),
        },
    }
    FAILURE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    display_cols = ["label", "cagr", "sharpe", "max_drawdown", "turnover", "filter", "fallback", "blend_spy"]
    print("top enhanced theme-index configs")
    print(enhanced[display_cols].head(15).to_string(index=False))
    print("\n2025Q2 rank-bucket diagnostics")
    print(
        rank_buckets[rank_buckets["period"].eq("2025Q2")][
            ["rank_bucket", "avg_fwd_5d_theme_return", "avg_fwd_5d_excess_vs_spy", "pct_negative_fwd_5d"]
        ].to_string(index=False)
    )
    print("\n2026YTD rank-bucket diagnostics")
    print(
        rank_buckets[rank_buckets["period"].eq("2026YTD")][
            ["rank_bucket", "avg_fwd_5d_theme_return", "avg_fwd_5d_excess_vs_spy", "pct_negative_fwd_5d"]
        ].to_string(index=False)
    )
    print(f"\nsaved failure/enhancement tests -> {OUT_DIR}")


if __name__ == "__main__":
    main()
