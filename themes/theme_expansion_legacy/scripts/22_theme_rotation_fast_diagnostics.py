from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

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
    load_theme_definitions,
    load_theme_memberships,
)


OUT_DIR = OUTPUT_DIR / "deep_theme_rotation"
STRATEGY_SWEEP_PATH = OUT_DIR / "strategy_sweep_results.csv"
WEAK_CONDITION_PATH = OUT_DIR / "weak_theme_condition_summary.csv"
WEAK_SEARCH_PATH = OUT_DIR / "weak_theme_condition_search.csv"
WEAK_THEME_STRATEGY_PATH = OUT_DIR / "weak_theme_index_strategy_results.csv"
YEARLY_COMPARISON_PATH = OUT_DIR / "yearly_spy_comparison.csv"
QUARTERLY_2025_PATH = OUT_DIR / "quarterly_2025_spy_comparison.csv"
UNIVERSE_AUDIT_PATH = OUT_DIR / "theme_universe_audit.csv"
OVERLAP_PATH = OUT_DIR / "theme_overlap_pairs.csv"
SUMMARY_PATH = OUT_DIR / "fast_deep_diagnostics_summary.json"

HOLD_DAYS = 5
TRADING_DAYS = 252.0


def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(THEME_SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    if "is_tradable" in scores.columns:
        scores = scores[scores["is_tradable"].fillna(False).astype(bool)].copy()
    scores = scores.dropna(subset=["theme_regime_rank"]).sort_values(["theme", "date"]).copy()
    scores["fwd_5d_theme_return"] = scores.groupby("theme")["theme_return_5d"].shift(-HOLD_DAYS)
    return scores.replace([np.inf, -np.inf], np.nan)


def load_returns() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    bars = bars.sort_values(["ticker", "date"])
    bars["return_1d"] = bars.groupby("ticker")["px"].pct_change()
    return bars.pivot(index="date", columns="ticker", values="return_1d").sort_index()


def perf_stats(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    returns = returns.fillna(0.0)
    if turnover is None:
        turnover = pd.Series(0.0, index=returns.index)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    years = max(len(returns) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 else np.nan
    loss_sum = abs(losers.sum())
    return {
        "total_return": final_equity - 1.0,
        "cagr": float(cagr) if pd.notna(cagr) else np.nan,
        "sharpe": float(np.sqrt(TRADING_DAYS) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
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
        rows.append(
            {
                "period": period,
                "strategy": label,
                "total_return": stats["total_return"],
                "max_drawdown": stats["max_drawdown"],
                "sharpe": stats["sharpe"],
                "hit_rate": stats["hit_rate"],
                "days": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_spy_comparison(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    q_pieces = []
    existing_path = BACKTEST_DIR / "rule_based_theme_rotation_daily.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        existing["date"] = pd.to_datetime(existing["date"])
        pieces.append(period_stats(existing[["date", "strategy_return"]], "existing_rule_based", "Y"))
        q_pieces.append(period_stats(existing[["date", "strategy_return"]], "existing_rule_based", "Q"))

    dates = existing["date"] if existing_path.exists() else pd.Series(returns.index)
    bench = returns.reindex(pd.to_datetime(dates))[[c for c in ["SPY", "QQQ"] if c in returns.columns]].fillna(0.0)
    bench = bench.reset_index().rename(columns={"index": "date"})
    for ticker in ["SPY", "QQQ"]:
        if ticker in bench:
            daily = bench[["date", ticker]].rename(columns={ticker: "strategy_return"})
            pieces.append(period_stats(daily, ticker, "Y"))
            q_pieces.append(period_stats(daily, ticker, "Q"))

    yearly = pd.concat(pieces, ignore_index=True)
    spy = yearly[yearly["strategy"].eq("SPY")][["period", "total_return"]].rename(columns={"total_return": "spy_return"})
    yearly = yearly.merge(spy, on="period", how="left")
    yearly["excess_vs_spy"] = yearly["total_return"] - yearly["spy_return"]

    quarterly = pd.concat(q_pieces, ignore_index=True)
    quarterly = quarterly[quarterly["period"].str.startswith("2025")].copy()
    spy_q = quarterly[quarterly["strategy"].eq("SPY")][["period", "total_return"]].rename(columns={"total_return": "spy_return"})
    quarterly = quarterly.merge(spy_q, on="period", how="left")
    quarterly["excess_vs_spy"] = quarterly["total_return"] - quarterly["spy_return"]
    return yearly, quarterly


def summarize_condition(data: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, object]:
    sample = data[mask & data["fwd_5d_theme_return"].notna()].copy()
    if sample.empty:
        return {"condition": name, "observations": 0}
    return {
        "condition": name,
        "observations": int(len(sample)),
        "avg_fwd_5d_theme_return": float(sample["fwd_5d_theme_return"].mean()),
        "median_fwd_5d_theme_return": float(sample["fwd_5d_theme_return"].median()),
        "pct_negative_fwd_5d": float((sample["fwd_5d_theme_return"] < 0).mean()),
        "avg_fwd_5d_excess_vs_spy": float(sample["fwd_5d_excess_vs_spy"].mean()),
        "short_ev_5d_before_cost": float(-sample["fwd_5d_theme_return"].mean()),
        "avg_rank": float(sample["theme_regime_rank"].mean()),
        "avg_theme_return_5d": float(sample["theme_return_5d"].mean()),
        "avg_theme_return_20d": float(sample["theme_return_20d"].mean()),
        "avg_above_20d_pct": float(sample["theme_above_20d_pct"].mean()),
        "avg_above_50d_pct": float(sample["theme_above_50d_pct"].mean()),
        "avg_rvol": float(sample["theme_rvol"].mean()),
    }


def build_weak_condition_tables(scores: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = scores.copy()
    spy_curve = (1.0 + returns["SPY"].fillna(0.0)).cumprod()
    spy_fwd = spy_curve.shift(-HOLD_DAYS) / spy_curve - 1.0
    data = data.merge(spy_fwd.rename("fwd_5d_spy_return").reset_index(), on="date", how="left")
    data["fwd_5d_excess_vs_spy"] = data["fwd_5d_theme_return"] - data["fwd_5d_spy_return"]
    daily_median_stability = data.groupby("date")["rank_stability_5d"].transform("median")

    masks = {
        "bottom_rank_76_plus": data["theme_regime_rank"] >= 76,
        "bottom_rank_86_plus": data["theme_regime_rank"] >= 86,
        "bottom_76_absolute_downtrend": (data["theme_regime_rank"] >= 76)
        & (data["theme_return_5d"] < 0)
        & (data["theme_return_20d"] < 0),
        "bottom_76_breakdown_confirmed": (data["theme_regime_rank"] >= 76)
        & (data["theme_return_5d"] < 0)
        & (data["theme_return_20d"] < 0)
        & (data["theme_vs_spy_20d"] < 0)
        & (data["theme_above_20d_pct"] < 0.50),
        "bottom_76_low_breadth_break": (data["theme_regime_rank"] >= 76)
        & (data["theme_above_20d_pct"] < 0.40)
        & (data["theme_above_50d_pct"] < 0.40),
        "bottom_76_high_volume_break": (data["theme_regime_rank"] >= 76)
        & (data["theme_return_5d"] < 0)
        & (data["theme_rvol"] > 1.20),
        "bottom_76_not_broken": (data["theme_regime_rank"] >= 76)
        & (data["theme_return_20d"] >= 0)
        & (data["theme_above_20d_pct"] >= 0.50),
        "bottom_76_rebound_setup": (data["theme_regime_rank"] >= 76)
        & (data["theme_return_5d"] < 0)
        & (data["theme_return_20d"] >= 0)
        & (data["theme_above_20d_pct"] >= 0.50),
        "bottom_76_internal_strength": (data["theme_regime_rank"] >= 76)
        & (data["theme_above_20d_pct"] >= 0.60),
        "bottom_76_stable_laggard": (data["theme_regime_rank"] >= 76)
        & (data["rank_stability_5d"] >= daily_median_stability),
    }
    summary = pd.DataFrame([summarize_condition(data, name, mask) for name, mask in masks.items()])

    predicates = {
        "ret5_lt0": data["theme_return_5d"] < 0,
        "ret5_lt_minus2": data["theme_return_5d"] < -0.02,
        "ret20_lt0": data["theme_return_20d"] < 0,
        "ret20_lt_minus5": data["theme_return_20d"] < -0.05,
        "vs_spy5_lt0": data["theme_vs_spy_5d"] < 0,
        "vs_spy20_lt0": data["theme_vs_spy_20d"] < 0,
        "above20_lt40": data["theme_above_20d_pct"] < 0.40,
        "above50_lt40": data["theme_above_50d_pct"] < 0.40,
        "rvol_gt120": data["theme_rvol"] > 1.20,
        "stability_gt_median": data["rank_stability_5d"] >= daily_median_stability,
    }
    rows = []
    for rank_floor in [51, 76, 86]:
        rank_mask = data["theme_regime_rank"] >= rank_floor
        for size in [1, 2, 3, 4]:
            for combo in combinations(predicates, size):
                mask = rank_mask.copy()
                for key in combo:
                    mask &= predicates[key]
                row = summarize_condition(data, f"rank>={rank_floor} & " + " & ".join(combo), mask)
                if row.get("observations", 0) >= 250:
                    rows.append(row)
    search = pd.DataFrame(rows)
    if not search.empty:
        search["short_quality_score"] = search["short_ev_5d_before_cost"] * search["pct_negative_fwd_5d"]
        search = search.sort_values(["short_ev_5d_before_cost", "pct_negative_fwd_5d"], ascending=False)
    return summary, search


def weak_mask(day: pd.DataFrame, condition: str) -> pd.Series:
    if condition == "rank_only":
        return day["theme_regime_rank"] >= 76
    if condition == "breakdown_confirmed":
        return (
            (day["theme_regime_rank"] >= 76)
            & (day["theme_return_5d"] < 0)
            & (day["theme_return_20d"] < 0)
            & (day["theme_vs_spy_20d"] < 0)
            & (day["theme_above_20d_pct"] < 0.50)
        )
    if condition == "low_breadth_break":
        return (day["theme_regime_rank"] >= 76) & (day["theme_above_20d_pct"] < 0.40) & (day["theme_above_50d_pct"] < 0.40)
    if condition == "not_broken":
        return (day["theme_regime_rank"] >= 76) & (day["theme_return_20d"] >= 0) & (day["theme_above_20d_pct"] >= 0.50)
    if condition == "rebound_setup":
        return (
            (day["theme_regime_rank"] >= 76)
            & (day["theme_return_5d"] < 0)
            & (day["theme_return_20d"] >= 0)
            & (day["theme_above_20d_pct"] >= 0.50)
        )
    raise ValueError(f"unknown condition {condition}")


def run_weak_theme_index_strategy(
    *,
    label: str,
    condition: str,
    direction: int,
    n_themes: int,
    scores: pd.DataFrame,
    theme_returns: pd.DataFrame,
) -> dict[str, object]:
    dates = sorted(set(scores["date"]).intersection(theme_returns.index))
    prev_weights = pd.Series(dtype=float)
    rows = []
    for date in dates:
        gross_return = (
            float((theme_returns.loc[date].reindex(prev_weights.index).fillna(0.0) * prev_weights).sum())
            if not prev_weights.empty
            else 0.0
        )
        day = scores[scores["date"].eq(date)].copy()
        day = day[weak_mask(day, condition)].sort_values("theme_regime_rank", ascending=False).head(n_themes)
        if day.empty:
            next_weights = pd.Series(dtype=float)
        else:
            weight = direction / len(day)
            next_weights = pd.Series(weight, index=day["theme"].tolist(), dtype=float)
        turnover = float(next_weights.sub(prev_weights, fill_value=0.0).abs().sum())
        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append({"date": date, "strategy_return": gross_return - cost, "turnover": turnover})
        prev_weights = next_weights
    daily = pd.DataFrame(rows)
    return {
        "label": label,
        "condition": condition,
        "direction": "short" if direction < 0 else "long",
        "n_themes": n_themes,
        **perf_stats(daily["strategy_return"], daily["turnover"]),
    }


def build_weak_strategy_results(scores: pd.DataFrame) -> pd.DataFrame:
    theme_daily = pd.read_parquet(THEME_DAILY_PATH)
    theme_daily["date"] = pd.to_datetime(theme_daily["date"])
    theme_daily = theme_daily[theme_daily["theme"].isin(scores["theme"].unique())].copy()
    theme_returns = theme_daily.pivot(index="date", columns="theme", values="theme_return_1d").sort_index()
    configs = [
        ("short_rank_only_bottom3", "rank_only", -1, 3),
        ("short_breakdown_bottom3", "breakdown_confirmed", -1, 3),
        ("short_breakdown_bottom5", "breakdown_confirmed", -1, 5),
        ("short_low_breadth_bottom3", "low_breadth_break", -1, 3),
        ("long_not_broken_bottom3", "not_broken", 1, 3),
        ("long_rebound_bottom3", "rebound_setup", 1, 3),
        ("long_rebound_bottom5", "rebound_setup", 1, 5),
    ]
    rows = [
        run_weak_theme_index_strategy(
            label=label,
            condition=condition,
            direction=direction,
            n_themes=n_themes,
            scores=scores,
            theme_returns=theme_returns,
        )
        for label, condition, direction, n_themes in configs
    ]
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def build_universe_audit(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    memberships = load_theme_memberships()
    defs = load_theme_definitions()
    latest_date = scores["date"].max()
    latest = scores[scores["date"].eq(latest_date)][
        [
            "theme",
            "theme_regime_rank",
            "theme_heat_rank",
            "theme_return_5d",
            "theme_vs_spy_20d",
            "theme_num_constituents",
            "theme_effective_constituents",
            "theme_effective_breadth",
        ]
    ].copy()
    participation = (
        scores.groupby("theme")
        .agg(
            days=("date", "size"),
            top5_days=("theme_regime_rank", lambda s: int((s <= 5).sum())),
            top10_days=("theme_regime_rank", lambda s: int((s <= 10).sum())),
            avg_rank=("theme_regime_rank", "mean"),
            avg_fwd_5d=("fwd_5d_theme_return", "mean"),
        )
        .reset_index()
    )
    counts = (
        memberships.groupby("theme")
        .agg(
            mapped_constituents=("ticker", "nunique"),
            category=("category", "first"),
            min_constituents=("min_constituents", "first"),
            max_constituents=("max_constituents", "first"),
            is_tradable=("is_tradable", "first"),
            is_watchlist_only=("is_watchlist_only", "first"),
            below_min=("is_below_min_constituents", "first"),
            tickers=("ticker", lambda s: "|".join(sorted(set(s))[:15])),
        )
        .reset_index()
    )
    audit = counts.merge(latest, on="theme", how="left").merge(participation, on="theme", how="left")
    audit["top5_pct"] = audit["top5_days"] / audit["days"].replace(0, np.nan)
    audit["live_testing_ready"] = (
        audit["is_tradable"].fillna(False).astype(bool)
        & ~audit["is_watchlist_only"].fillna(False).astype(bool)
        & ~audit["below_min"].fillna(False).astype(bool)
        & audit["theme_regime_rank"].notna()
    )
    ai_keywords = [
        "ai",
        "cloud",
        "data",
        "datacenter",
        "semi",
        "memory",
        "networking",
        "optical",
        "power_grid",
        "electrification",
        "nuclear",
        "uranium",
        "copper",
        "rare_earth",
        "critical_minerals",
        "robotics",
        "automation",
        "quantum",
        "space",
    ]
    audit["ai_buildout_related"] = audit["theme"].str.contains("|".join(ai_keywords), case=False, regex=True)
    audit["readiness_bucket"] = np.select(
        [
            audit["live_testing_ready"] & (audit["mapped_constituents"] >= 8),
            audit["live_testing_ready"] & (audit["mapped_constituents"] < 8),
            audit["is_watchlist_only"].fillna(False).astype(bool),
        ],
        ["ready_core", "ready_thin", "watchlist_only"],
        default="needs_review",
    )
    audit = audit.sort_values(["live_testing_ready", "theme_regime_rank"], ascending=[False, True])

    theme_sets = memberships.groupby("theme")["ticker"].apply(lambda s: set(s)).to_dict()
    rows = []
    themes = sorted(theme_sets)
    for idx, left in enumerate(themes):
        for right in themes[idx + 1 :]:
            a = theme_sets[left]
            b = theme_sets[right]
            intersection = len(a & b)
            if not intersection:
                continue
            jaccard = intersection / len(a | b)
            overlap_coefficient = intersection / min(len(a), len(b))
            if jaccard >= 0.25 or overlap_coefficient >= 0.50:
                rows.append(
                    {
                        "theme_a": left,
                        "theme_b": right,
                        "intersection": intersection,
                        "jaccard": jaccard,
                        "overlap_coefficient": overlap_coefficient,
                    }
                )
    overlaps = pd.DataFrame(rows)
    if not overlaps.empty:
        overlaps = overlaps.sort_values(["overlap_coefficient", "jaccard"], ascending=False)

    summary = {
        "latest_date": str(pd.Timestamp(latest_date).date()),
        "theme_definitions": int(len(defs)),
        "mapped_themes": int(audit["theme"].nunique()),
        "live_testing_ready_themes": int(audit["live_testing_ready"].sum()),
        "watchlist_only_themes": int(audit["is_watchlist_only"].fillna(False).sum()),
        "ready_thin_themes": int(audit["readiness_bucket"].eq("ready_thin").sum()),
        "median_mapped_constituents": float(audit["mapped_constituents"].median()),
        "ai_buildout_related_ready_themes": int((audit["ai_buildout_related"] & audit["live_testing_ready"]).sum()),
        "readiness_counts": audit["readiness_bucket"].value_counts().to_dict(),
        "latest_top20_themes": audit[audit["live_testing_ready"]].nsmallest(20, "theme_regime_rank")["theme"].tolist(),
        "largest_overlap_pairs": overlaps.head(15).to_dict(orient="records") if not overlaps.empty else [],
    }
    return audit, overlaps, summary


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    returns = load_returns()

    yearly, quarterly = build_spy_comparison(returns)
    yearly.to_csv(YEARLY_COMPARISON_PATH, index=False)
    quarterly.to_csv(QUARTERLY_2025_PATH, index=False)

    weak_summary, weak_search = build_weak_condition_tables(scores, returns)
    weak_summary.to_csv(WEAK_CONDITION_PATH, index=False)
    weak_search.to_csv(WEAK_SEARCH_PATH, index=False)
    weak_strategy = build_weak_strategy_results(scores)
    weak_strategy.to_csv(WEAK_THEME_STRATEGY_PATH, index=False)

    audit, overlaps, universe_summary = build_universe_audit(scores)
    audit.to_csv(UNIVERSE_AUDIT_PATH, index=False)
    overlaps.to_csv(OVERLAP_PATH, index=False)

    strategy_sweep = pd.read_csv(STRATEGY_SWEEP_PATH) if STRATEGY_SWEEP_PATH.exists() else pd.DataFrame()
    summary = {
        "strategy_sweep_rows_available": int(len(strategy_sweep)),
        "top_strategy_rows_available": strategy_sweep.sort_values("cagr", ascending=False).head(15).to_dict(orient="records")
        if not strategy_sweep.empty
        else [],
        "yearly_2025": yearly[yearly["period"].eq("2025")].to_dict(orient="records"),
        "weak_condition_summary": weak_summary.to_dict(orient="records"),
        "top_short_condition_search": weak_search.head(15).to_dict(orient="records") if not weak_search.empty else [],
        "weak_theme_index_strategy_results": weak_strategy.to_dict(orient="records"),
        "universe_summary": universe_summary,
        "output_files": {
            "strategy_sweep": str(STRATEGY_SWEEP_PATH),
            "yearly_comparison": str(YEARLY_COMPARISON_PATH),
            "quarterly_2025": str(QUARTERLY_2025_PATH),
            "weak_conditions": str(WEAK_CONDITION_PATH),
            "weak_condition_search": str(WEAK_SEARCH_PATH),
            "weak_theme_index_strategies": str(WEAK_THEME_STRATEGY_PATH),
            "universe_audit": str(UNIVERSE_AUDIT_PATH),
            "overlaps": str(OVERLAP_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("2025 comparison")
    print(yearly[yearly["period"].eq("2025")].to_string(index=False))
    print("\nweak condition summary")
    print(
        weak_summary[
            [
                "condition",
                "observations",
                "avg_fwd_5d_theme_return",
                "pct_negative_fwd_5d",
                "short_ev_5d_before_cost",
            ]
        ].to_string(index=False)
    )
    print("\nweak theme index strategies")
    print(weak_strategy[["label", "direction", "cagr", "sharpe", "max_drawdown", "turnover"]].to_string(index=False))
    print(f"\nsaved fast diagnostics -> {OUT_DIR}")


if __name__ == "__main__":
    main()
