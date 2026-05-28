from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
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
WEAK_STRATEGY_PATH = OUT_DIR / "weak_theme_strategy_results.csv"
YEARLY_COMPARISON_PATH = OUT_DIR / "yearly_spy_comparison.csv"
QUARTERLY_COMPARISON_PATH = OUT_DIR / "quarterly_2025_spy_comparison.csv"
UNIVERSE_AUDIT_PATH = OUT_DIR / "theme_universe_audit.csv"
OVERLAP_PATH = OUT_DIR / "theme_overlap_pairs.csv"
SUMMARY_PATH = OUT_DIR / "deep_theme_rotation_summary.json"
FRONTIER_PLOT_PATH = OUT_DIR / "strategy_frontier.png"

HOLD_DAYS = 5
TRADING_DAYS = 252.0


BASKET_DEFS: dict[str, list[tuple[int, int, float]]] = {
    "leaders_1_3": [(1, 3, 1.0)],
    "leaders_1_6": [(1, 6, 1.0)],
    "leaders_1_9": [(1, 9, 1.0)],
    "followers_4_6": [(4, 6, 1.0)],
    "followers_4_9": [(4, 9, 1.0)],
    "followers_7_15": [(7, 15, 1.0)],
    "laggards_10_20": [(10, 20, 1.0)],
    "blend_1_3_4_9_50_50": [(1, 3, 0.5), (4, 9, 0.5)],
    "blend_1_3_4_9_33_67": [(1, 3, 0.33), (4, 9, 0.67)],
}


def load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


leader_builder = load_script("05_rank_theme_leaders.py", "leader_builder_deep")


def clean_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(THEME_SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    if "is_tradable" in scores.columns:
        scores = scores[scores["is_tradable"].fillna(False).astype(bool)].copy()
    scores = scores.dropna(subset=["theme_regime_rank"]).copy()
    scores = scores.sort_values(["theme", "date"])
    scores["fwd_5d_theme_return"] = scores.groupby("theme")["theme_return_5d"].shift(-HOLD_DAYS)

    lower_is_better = scores.groupby("date")["theme_regime_rank"].rank(pct=True, ascending=True)
    heat_lower_is_better = scores.groupby("date")["theme_heat_rank"].rank(pct=True, ascending=True)
    if "rank_stability_5d" in scores.columns:
        stable_lower_is_better = scores.groupby("date")["rank_stability_5d"].rank(pct=True, ascending=False)
    else:
        stable_lower_is_better = 0.5
    scores["theme_composite_score_deep"] = (
        0.55 * lower_is_better.fillna(0.5)
        + 0.25 * heat_lower_is_better.fillna(0.5)
        + 0.20 * pd.Series(stable_lower_is_better, index=scores.index).fillna(0.5)
    )
    scores["theme_composite_rank"] = scores.groupby("date")["theme_composite_score_deep"].rank(
        ascending=True,
        method="first",
    )
    scores["oracle_theme_rank"] = scores.groupby("date")["fwd_5d_theme_return"].rank(
        ascending=False,
        method="first",
    )
    return scores.replace([np.inf, -np.inf], np.nan)


def load_stock_returns_and_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    bars = bars.sort_values(["ticker", "date"])
    bars["stock_return_1d"] = bars.groupby("ticker")["px"].pct_change()
    returns = bars.pivot(index="date", columns="ticker", values="stock_return_1d").sort_index()
    prices = bars.pivot(index="date", columns="ticker", values="px").sort_index()
    return returns, prices


def build_local_market_regime(prices: pd.DataFrame) -> pd.Series:
    out = pd.DataFrame(index=prices.index)
    for ticker in ["SPY", "QQQ"]:
        if ticker in prices:
            out[f"{ticker.lower()}_above_200dma"] = prices[ticker] > prices[ticker].rolling(200, min_periods=100).mean()
        else:
            out[f"{ticker.lower()}_above_200dma"] = False

    excluded = {"SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "SMH", "XLK", "XLF", "XLV", "XLE"}
    stock_cols = [col for col in prices.columns if col not in excluded]
    sma200 = prices[stock_cols].rolling(200, min_periods=100).mean()
    out["market_breadth"] = (prices[stock_cols] > sma200).mean(axis=1)
    risk_off = (
        (~out["spy_above_200dma"].fillna(False) & ~out["qqq_above_200dma"].fillna(False))
        | (out["market_breadth"] < 0.40)
    )
    risk_on = (
        out["spy_above_200dma"].fillna(False)
        & out["qqq_above_200dma"].fillna(False)
        & (out["market_breadth"] > 0.55)
    )
    return pd.Series(np.select([risk_on, risk_off], ["risk_on", "risk_off"], default="neutral"), index=prices.index)


def exposure_for(mode: str, regime: str) -> float:
    if mode == "full":
        return 1.0
    if mode == "scaled":
        return {"risk_on": 1.0, "neutral": 0.70, "risk_off": 0.30}.get(regime, 0.70)
    if mode == "cash_risk_off":
        return {"risk_on": 1.0, "neutral": 0.70, "risk_off": 0.0}.get(regime, 0.70)
    raise ValueError(f"unknown exposure mode {mode}")


def build_all_member_panel(scores: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    panel = leader_builder.load_stock_theme_panel()
    panel = panel.merge(scores[["date", "theme", "theme_return_5d"]], on=["date", "theme"], how="inner")
    panel["stock_vs_theme_5d"] = panel["stock_return_5d"] - panel["theme_return_5d"]
    panel = leader_builder.add_leader_follower_scores(panel)
    panel["leader_rank"] = panel["ticker_rank_in_theme"]

    curve = (1.0 + returns.fillna(0.0)).cumprod()
    fwd_5d = curve.shift(-HOLD_DAYS) / curve - 1.0
    fwd_long = fwd_5d.stack().rename("fwd_5d_stock_return").reset_index()
    fwd_long = fwd_long.rename(columns={"level_1": "ticker"})
    panel = panel.merge(fwd_long, on=["date", "ticker"], how="left")
    return panel.replace([np.inf, -np.inf], np.nan)


def select_members(day_panel: pd.DataFrame, theme: str, basket: str) -> pd.Series:
    theme_rows = day_panel[day_panel["theme"].eq(theme)].copy()
    if theme_rows.empty:
        return pd.Series(dtype=float)

    weights: dict[str, float] = {}
    if basket == "oracle_stock_top3":
        selected = theme_rows.dropna(subset=["fwd_5d_stock_return"]).nlargest(3, "fwd_5d_stock_return")
        if selected.empty:
            return pd.Series(dtype=float)
        tickers = selected["ticker"].dropna().tolist()
        for ticker in tickers:
            weights[ticker] = weights.get(ticker, 0.0) + 1.0 / len(tickers)
    else:
        for min_rank, max_rank, sleeve_weight in BASKET_DEFS[basket]:
            selected = theme_rows[theme_rows["leader_rank"].between(min_rank, max_rank)].sort_values("leader_rank")
            tickers = selected["ticker"].dropna().tolist()
            if not tickers:
                continue
            for ticker in tickers:
                weights[ticker] = weights.get(ticker, 0.0) + sleeve_weight / len(tickers)

    out = pd.Series(weights, dtype=float)
    total = out.abs().sum()
    return out / total if total else pd.Series(dtype=float)


def perf_stats(returns: pd.Series, turnover: pd.Series) -> dict[str, float]:
    returns = returns.fillna(0.0)
    turnover = turnover.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    loss_sum = abs(losers.sum())
    equity_final = float(equity.iloc[-1]) if len(equity) else 1.0
    cagr = equity_final ** (1.0 / years) - 1.0 if equity_final > 0 else np.nan
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    return {
        "total_return": equity_final - 1.0,
        "cagr": float(cagr) if pd.notna(cagr) else np.nan,
        "sharpe": float(np.sqrt(TRADING_DAYS) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "sortino": float(np.sqrt(TRADING_DAYS) * returns.mean() / losers.std()) if losers.std() else 0.0,
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if pd.notna(cagr) and max_drawdown < 0 else np.nan,
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else np.nan,
        "hit_rate": float((returns > 0).mean()),
        "avg_daily_return": float(returns.mean()),
        "daily_vol": float(returns.std()),
        "turnover": float(turnover.mean()),
    }


def run_theme_rotation(
    cfg: dict[str, object],
    *,
    dates: list[pd.Timestamp],
    scores: pd.DataFrame,
    scores_by_date: dict[str, dict[pd.Timestamp, pd.DataFrame]],
    rank_lookup: dict[str, pd.Series],
    panel_by_date: dict[pd.Timestamp, pd.DataFrame],
    returns: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    rank_col = str(cfg["rank_col"])
    top_n = int(cfg["top_n"])
    exit_rank = int(cfg["exit_rank"])
    min_hold = int(cfg["min_hold"])
    basket = str(cfg["basket"])
    exposure_mode = str(cfg["exposure_mode"])
    cap_active = bool(cfg.get("cap_active", False))

    active: dict[str, dict[str, object]] = {}
    prev_weights = pd.Series(dtype=float)
    rows: list[dict[str, object]] = []

    for i, date in enumerate(dates):
        gross_return = (
            float((returns.loc[date].reindex(prev_weights.index).fillna(0.0) * prev_weights).sum())
            if not prev_weights.empty
            else 0.0
        )

        updated = dict(active)
        lookup = rank_lookup[rank_col]
        for theme, state in list(active.items()):
            held_days = i - int(state["start_i"]) + 1
            rank = lookup.get((date, theme), np.nan)
            if held_days >= min_hold and (pd.isna(rank) or rank > exit_rank):
                updated.pop(theme, None)

        day_scores = scores_by_date[rank_col].get(date)
        day_panel = panel_by_date.get(date)
        if day_scores is not None and day_panel is not None:
            entries = day_scores[day_scores[rank_col].le(top_n)].head(top_n)
            for entry in entries.itertuples(index=False):
                theme = str(entry.theme)
                if theme in updated:
                    continue
                member_weights = select_members(day_panel, theme, basket)
                member_weights = member_weights[member_weights.index.isin(returns.columns)]
                if member_weights.empty:
                    continue
                updated[theme] = {
                    "start_i": i + 1,
                    "member_weights": member_weights,
                }

            if cap_active and len(updated) > top_n:
                ranked_active = sorted(
                    updated,
                    key=lambda theme: lookup.get((date, theme), np.inf),
                )
                updated = {theme: updated[theme] for theme in ranked_active[:top_n]}

        regime = regimes.get(date, "neutral")
        target_exposure = exposure_for(exposure_mode, str(regime))
        if updated and target_exposure > 0:
            theme_weight = target_exposure / len(updated)
            pieces = []
            for state in updated.values():
                weights = state["member_weights"] * theme_weight
                pieces.append(weights)
            after_trade = pd.concat(pieces, axis=1).sum(axis=1) if pieces else pd.Series(dtype=float)
        else:
            after_trade = pd.Series(dtype=float)

        turnover = float(after_trade.sub(prev_weights, fill_value=0.0).abs().sum())
        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append(
            {
                "date": date,
                "strategy_return": gross_return - cost,
                "gross_return": gross_return,
                "turnover": turnover,
                "exposure": float(after_trade.abs().sum()) if not after_trade.empty else 0.0,
                "n_positions": int(len(after_trade)),
                "n_active_themes": int(len(updated)),
                "regime": regime,
                "active_themes": "|".join(sorted(updated)),
            }
        )
        active = updated
        prev_weights = after_trade

    out = pd.DataFrame(rows)
    out["equity"] = (1.0 + out["strategy_return"].fillna(0.0)).cumprod()
    return out


def add_config(configs: list[dict[str, object]], seen: set[str], **kwargs: object) -> None:
    label = clean_label(
        f"{kwargs['rank_col']}__{kwargs['basket']}__top{kwargs['top_n']}"
        f"_exit{kwargs['exit_rank']}_hold{kwargs['min_hold']}_{kwargs['exposure_mode']}"
        f"{'_cap' if kwargs.get('cap_active') else ''}"
    )
    if label in seen:
        return
    cfg = dict(kwargs)
    cfg["label"] = label
    cfg["is_oracle"] = str(cfg["rank_col"]).startswith("oracle") or str(cfg["basket"]).startswith("oracle")
    configs.append(cfg)
    seen.add(label)


def build_strategy_grid() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    seen: set[str] = set()

    # A targeted grid: broad enough to compare construction choices, small enough
    # to rerun interactively on the local parquet dataset.
    for basket in BASKET_DEFS:
        add_config(
            configs,
            seen,
            rank_col="theme_regime_rank",
            basket=basket,
            top_n=3,
            exit_rank=12,
            min_hold=5,
            exposure_mode="full",
            cap_active=False,
        )

    for basket in ["leaders_1_3", "followers_4_9", "blend_1_3_4_9_50_50"]:
        for top_n in [1, 2, 3, 5, 8]:
            add_config(
                configs,
                seen,
                rank_col="theme_regime_rank",
                basket=basket,
                top_n=top_n,
                exit_rank=12,
                min_hold=5,
                exposure_mode="full",
                cap_active=False,
            )

    for exit_rank in [8, 12, 20]:
        for min_hold in [1, 5, 10]:
            add_config(
                configs,
                seen,
                rank_col="theme_regime_rank",
                basket="blend_1_3_4_9_50_50",
                top_n=3,
                exit_rank=exit_rank,
                min_hold=min_hold,
                exposure_mode="full",
                cap_active=False,
            )

    for basket in ["leaders_1_3", "followers_4_9", "blend_1_3_4_9_50_50"]:
        for rank_col in ["theme_regime_rank", "theme_heat_rank", "theme_composite_rank"]:
            for exposure_mode in ["full", "scaled"]:
                add_config(
                    configs,
                    seen,
                    rank_col=rank_col,
                    basket=basket,
                    top_n=3,
                    exit_rank=12,
                    min_hold=5,
                    exposure_mode=exposure_mode,
                    cap_active=False,
                )

    for basket in ["leaders_1_3", "blend_1_3_4_9_50_50"]:
        for top_n in [3, 5]:
            add_config(
                configs,
                seen,
                rank_col="theme_regime_rank",
                basket=basket,
                top_n=top_n,
                exit_rank=12,
                min_hold=5,
                exposure_mode="full",
                cap_active=True,
            )

    for rank_col in ["oracle_theme_rank", "theme_regime_rank"]:
        for basket in ["leaders_1_3", "leaders_1_9", "blend_1_3_4_9_50_50", "oracle_stock_top3"]:
            add_config(
                configs,
                seen,
                rank_col=rank_col,
                basket=basket,
                top_n=3,
                exit_rank=12,
                min_hold=1,
                exposure_mode="full",
                cap_active=False,
            )

    return configs


def benchmark_returns(returns: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    bench = returns.reindex(dates)[[col for col in ["SPY", "QQQ"] if col in returns.columns]].fillna(0.0)
    return bench


def period_stats(frame: pd.DataFrame, label: str, freq: str = "Y") -> pd.DataFrame:
    out = frame.copy()
    out["period"] = out["date"].dt.to_period(freq).astype(str)
    rows = []
    for period, group in out.groupby("period", sort=True):
        returns = group["strategy_return"].fillna(0.0)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rows.append(
            {
                "period": period,
                "strategy": label,
                "total_return": float(equity.iloc[-1] - 1.0),
                "max_drawdown": float(drawdown.min()),
                "sharpe": float(np.sqrt(TRADING_DAYS) * returns.mean() / returns.std()) if returns.std() else 0.0,
                "hit_rate": float((returns > 0).mean()),
                "days": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_yearly_comparison(
    top_daily: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    for label, daily in top_daily.items():
        pieces.append(period_stats(daily[["date", "strategy_return"]], label, "Y"))

    bench = benchmark_returns(returns, dates).reset_index().rename(columns={"index": "date"})
    for ticker in ["SPY", "QQQ"]:
        if ticker in bench:
            pieces.append(period_stats(bench[["date", ticker]].rename(columns={ticker: "strategy_return"}), ticker, "Y"))

    yearly = pd.concat(pieces, ignore_index=True)
    spy = yearly[yearly["strategy"].eq("SPY")][["period", "total_return"]].rename(columns={"total_return": "spy_return"})
    yearly = yearly.merge(spy, on="period", how="left")
    yearly["excess_vs_spy"] = yearly["total_return"] - yearly["spy_return"]

    quarterly_pieces = []
    for label, daily in top_daily.items():
        quarterly_pieces.append(period_stats(daily[["date", "strategy_return"]], label, "Q"))
    for ticker in ["SPY", "QQQ"]:
        if ticker in bench:
            quarterly_pieces.append(period_stats(bench[["date", ticker]].rename(columns={ticker: "strategy_return"}), ticker, "Q"))
    quarterly = pd.concat(quarterly_pieces, ignore_index=True)
    quarterly = quarterly[quarterly["period"].str.startswith("2025")].copy()
    spy_q = quarterly[quarterly["strategy"].eq("SPY")][["period", "total_return"]].rename(columns={"total_return": "spy_return"})
    quarterly = quarterly.merge(spy_q, on="period", how="left")
    quarterly["excess_vs_spy"] = quarterly["total_return"] - quarterly["spy_return"]
    return yearly, quarterly


def summarize_condition(data: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, object]:
    sample = data[mask & data["fwd_5d_theme_return"].notna()].copy()
    if sample.empty:
        return {
            "condition": name,
            "observations": 0,
            "avg_fwd_5d_theme_return": np.nan,
            "median_fwd_5d_theme_return": np.nan,
            "pct_negative_fwd_5d": np.nan,
            "avg_fwd_5d_excess_vs_spy": np.nan,
            "short_ev_5d_before_cost": np.nan,
        }
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
        "avg_rvol": float(sample["theme_rvol"].mean()),
    }


def build_weak_theme_diagnostics(scores: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = scores.copy()
    curve = (1.0 + returns["SPY"].fillna(0.0)).cumprod() if "SPY" in returns else pd.Series(dtype=float)
    if not curve.empty:
        spy_fwd = curve.shift(-HOLD_DAYS) / curve - 1.0
        data = data.merge(spy_fwd.rename("fwd_5d_spy_return").reset_index(), on="date", how="left")
    else:
        data["fwd_5d_spy_return"] = np.nan
    data["fwd_5d_excess_vs_spy"] = data["fwd_5d_theme_return"] - data["fwd_5d_spy_return"]
    daily_median_stability = data.groupby("date")["rank_stability_5d"].transform("median")

    condition_masks = {
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
        "bottom_76_high_volume_break": (data["theme_regime_rank"] >= 76)
        & (data["theme_return_5d"] < 0)
        & (data["theme_rvol"] > 1.20),
        "bottom_76_low_breadth_break": (data["theme_regime_rank"] >= 76)
        & (data["theme_above_20d_pct"] < 0.40)
        & (data["theme_above_50d_pct"] < 0.40),
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
    summary = pd.DataFrame([summarize_condition(data, name, mask) for name, mask in condition_masks.items()])

    predicates: dict[str, pd.Series] = {
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
    search_rows = []
    for rank_floor in [51, 76, 86]:
        rank_mask = data["theme_regime_rank"] >= rank_floor
        for combo_len in [1, 2, 3, 4]:
            for combo in combinations(predicates, combo_len):
                mask = rank_mask.copy()
                for key in combo:
                    mask &= predicates[key]
                row = summarize_condition(data, f"rank>={rank_floor} & " + " & ".join(combo), mask)
                if row["observations"] >= 250:
                    search_rows.append(row)
    search = pd.DataFrame(search_rows)
    if not search.empty:
        search["short_quality_score"] = search["short_ev_5d_before_cost"] * search["pct_negative_fwd_5d"]
        search = search.sort_values(["short_ev_5d_before_cost", "pct_negative_fwd_5d"], ascending=False)
    return summary, search


def weak_condition_mask(scores: pd.DataFrame, name: str) -> pd.Series:
    if name == "rank_only":
        return scores["theme_regime_rank"] >= 76
    if name == "breakdown_confirmed":
        return (
            (scores["theme_regime_rank"] >= 76)
            & (scores["theme_return_5d"] < 0)
            & (scores["theme_return_20d"] < 0)
            & (scores["theme_vs_spy_20d"] < 0)
            & (scores["theme_above_20d_pct"] < 0.50)
        )
    if name == "high_volume_break":
        return (scores["theme_regime_rank"] >= 76) & (scores["theme_return_5d"] < 0) & (scores["theme_rvol"] > 1.20)
    if name == "not_broken":
        return (
            (scores["theme_regime_rank"] >= 76)
            & (scores["theme_return_20d"] >= 0)
            & (scores["theme_above_20d_pct"] >= 0.50)
        )
    if name == "rebound_setup":
        return (
            (scores["theme_regime_rank"] >= 76)
            & (scores["theme_return_5d"] < 0)
            & (scores["theme_return_20d"] >= 0)
            & (scores["theme_above_20d_pct"] >= 0.50)
        )
    raise ValueError(f"unknown weak condition {name}")


def select_weak_members(day_panel: pd.DataFrame, theme: str, mode: str) -> list[str]:
    rows = day_panel[day_panel["theme"].eq(theme)].copy()
    if rows.empty:
        return []
    if mode == "weakest3":
        return rows.nlargest(3, "leader_rank")["ticker"].dropna().tolist()
    if mode == "leaders3":
        return rows.nsmallest(3, "leader_rank")["ticker"].dropna().tolist()
    if mode == "all_equal":
        return rows["ticker"].dropna().tolist()
    raise ValueError(f"unknown weak member mode {mode}")


def run_weak_strategy(
    *,
    label: str,
    condition: str,
    direction: int,
    member_mode: str,
    n_themes: int,
    scores: pd.DataFrame,
    dates: list[pd.Timestamp],
    panel_by_date: dict[pd.Timestamp, pd.DataFrame],
    returns: pd.DataFrame,
) -> dict[str, object]:
    prev_weights = pd.Series(dtype=float)
    rows = []
    for date in dates:
        gross_return = (
            float((returns.loc[date].reindex(prev_weights.index).fillna(0.0) * prev_weights).sum())
            if not prev_weights.empty
            else 0.0
        )
        day_scores = scores[scores["date"].eq(date)].copy()
        day_scores = day_scores[weak_condition_mask(day_scores, condition)]
        if direction < 0:
            day_scores = day_scores.sort_values("theme_regime_rank", ascending=False).head(n_themes)
        else:
            day_scores = day_scores.sort_values("theme_regime_rank", ascending=False).head(n_themes)

        weights: dict[str, float] = {}
        day_panel = panel_by_date.get(date)
        if day_panel is not None and not day_scores.empty:
            for theme in day_scores["theme"]:
                tickers = select_weak_members(day_panel, str(theme), member_mode)
                tickers = [ticker for ticker in tickers if ticker in returns.columns]
                if not tickers:
                    continue
                per_theme_weight = direction * (1.0 / len(day_scores))
                for ticker in tickers:
                    weights[ticker] = weights.get(ticker, 0.0) + per_theme_weight / len(tickers)
        after_trade = pd.Series(weights, dtype=float)
        turnover = float(after_trade.sub(prev_weights, fill_value=0.0).abs().sum())
        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append({"date": date, "strategy_return": gross_return - cost, "turnover": turnover})
        prev_weights = after_trade

    daily = pd.DataFrame(rows)
    stats = perf_stats(daily["strategy_return"], daily["turnover"])
    return {"label": label, "condition": condition, "direction": "short" if direction < 0 else "long", "member_mode": member_mode, "n_themes": n_themes, **stats}


def run_weak_strategy_suite(
    scores: pd.DataFrame,
    dates: list[pd.Timestamp],
    panel_by_date: dict[pd.Timestamp, pd.DataFrame],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    configs = [
        ("short_rank_only_weakest3", "rank_only", -1, "weakest3", 3),
        ("short_rank_only_leaders3", "rank_only", -1, "leaders3", 3),
        ("short_breakdown_confirmed_weakest3", "breakdown_confirmed", -1, "weakest3", 3),
        ("short_breakdown_confirmed_leaders3", "breakdown_confirmed", -1, "leaders3", 3),
        ("short_high_volume_break_weakest3", "high_volume_break", -1, "weakest3", 3),
        ("long_not_broken_leaders3", "not_broken", 1, "leaders3", 3),
        ("long_rebound_setup_leaders3", "rebound_setup", 1, "leaders3", 3),
        ("long_rebound_setup_weakest3", "rebound_setup", 1, "weakest3", 3),
    ]
    rows = [
        run_weak_strategy(
            label=label,
            condition=condition,
            direction=direction,
            member_mode=member_mode,
            n_themes=n_themes,
            scores=scores,
            dates=dates,
            panel_by_date=panel_by_date,
            returns=returns,
        )
        for label, condition, direction, member_mode, n_themes in configs
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
    overlap_rows = []
    themes = sorted(theme_sets)
    for i, left in enumerate(themes):
        for right in themes[i + 1 :]:
            a = theme_sets[left]
            b = theme_sets[right]
            if not a or not b:
                continue
            intersection = len(a & b)
            if not intersection:
                continue
            union = len(a | b)
            jaccard = intersection / union
            overlap_coeff = intersection / min(len(a), len(b))
            if jaccard >= 0.25 or overlap_coeff >= 0.50:
                overlap_rows.append(
                    {
                        "theme_a": left,
                        "theme_b": right,
                        "intersection": intersection,
                        "jaccard": jaccard,
                        "overlap_coefficient": overlap_coeff,
                    }
                )
    overlaps = pd.DataFrame(overlap_rows)
    if not overlaps.empty:
        overlaps = overlaps.sort_values(["overlap_coefficient", "jaccard"], ascending=False)

    summary = {
        "latest_date": str(pd.Timestamp(latest_date).date()),
        "theme_definitions": int(len(defs)),
        "mapped_themes": int(audit["theme"].nunique()),
        "tradable_ready_themes": int(audit["live_testing_ready"].sum()),
        "watchlist_only_themes": int(audit["is_watchlist_only"].fillna(False).sum()),
        "ready_thin_themes": int(audit["readiness_bucket"].eq("ready_thin").sum()),
        "median_mapped_constituents": float(audit["mapped_constituents"].median()),
        "ai_buildout_related_ready_themes": int((audit["ai_buildout_related"] & audit["live_testing_ready"]).sum()),
        "readiness_counts": audit["readiness_bucket"].value_counts().to_dict(),
        "category_counts": audit[audit["live_testing_ready"]]["category"].value_counts().to_dict(),
        "latest_top20_themes": audit[audit["live_testing_ready"]].nsmallest(20, "theme_regime_rank")["theme"].tolist(),
        "largest_overlap_pairs": overlaps.head(15).to_dict(orient="records") if not overlaps.empty else [],
    }
    return audit, overlaps, summary


def plot_frontier(results: pd.DataFrame) -> None:
    non_oracle = results[~results["is_oracle"].fillna(False)].copy()
    if non_oracle.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    baskets = sorted(non_oracle["basket"].unique())
    for basket in baskets:
        sample = non_oracle[non_oracle["basket"].eq(basket)]
        ax.scatter(
            sample["max_drawdown"].abs(),
            sample["cagr"],
            s=22,
            alpha=0.55,
            label=basket,
        )
    ax.set_title("Theme Rotation Strategy Frontier")
    ax.set_xlabel("Absolute max drawdown")
    ax.set_ylabel("CAGR")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(FRONTIER_PLOT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    returns, prices = load_stock_returns_and_prices()
    regimes = build_local_market_regime(prices)
    dates = sorted(set(scores["date"]).intersection(returns.index))

    print(f"loaded {scores['theme'].nunique()} tradable themes across {len(dates)} trading days", flush=True)
    print("building full leader/follower member panel...", flush=True)
    panel = build_all_member_panel(scores, returns)
    panel_by_date = {date: frame for date, frame in panel.groupby("date", sort=False)}
    rank_cols = ["theme_regime_rank", "theme_heat_rank", "theme_composite_rank", "oracle_theme_rank"]
    scores_by_date = {
        col: {
            date: frame.dropna(subset=[col]).sort_values(col)
            for date, frame in scores.groupby("date", sort=False)
        }
        for col in rank_cols
    }
    rank_lookup = {col: scores.set_index(["date", "theme"])[col] for col in rank_cols}

    configs = build_strategy_grid()
    rows = []
    print(f"running {len(configs)} strategy configurations...", flush=True)
    for idx, cfg in enumerate(configs, start=1):
        daily = run_theme_rotation(
            cfg,
            dates=dates,
            scores=scores,
            scores_by_date=scores_by_date,
            rank_lookup=rank_lookup,
            panel_by_date=panel_by_date,
            returns=returns,
            regimes=regimes,
        )
        stats = perf_stats(daily["strategy_return"], daily["turnover"])
        rows.append({**cfg, **stats})
        if idx % 10 == 0:
            pd.DataFrame(rows).to_csv(STRATEGY_SWEEP_PATH, index=False)
        if idx % 10 == 0:
            print(f"  finished {idx}/{len(configs)}", flush=True)

    results = pd.DataFrame(rows).sort_values(["is_oracle", "cagr", "sharpe"], ascending=[True, False, False])
    results.to_csv(STRATEGY_SWEEP_PATH, index=False)
    plot_frontier(results)

    best_cagr_cfg = results[~results["is_oracle"]].sort_values("cagr", ascending=False).iloc[0].to_dict()
    best_sharpe_cfg = results[~results["is_oracle"]].sort_values("sharpe", ascending=False).iloc[0].to_dict()
    top_daily: dict[str, pd.DataFrame] = {}
    for label, cfg in [("deep_best_cagr", best_cagr_cfg), ("deep_best_sharpe", best_sharpe_cfg)]:
        daily = run_theme_rotation(
            cfg,
            dates=dates,
            scores=scores,
            scores_by_date=scores_by_date,
            rank_lookup=rank_lookup,
            panel_by_date=panel_by_date,
            returns=returns,
            regimes=regimes,
        )
        daily.to_parquet(OUT_DIR / f"{label}_daily.parquet", index=False)
        top_daily[label] = daily

    existing_path = BACKTEST_DIR / "rule_based_theme_rotation_daily.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        existing["date"] = pd.to_datetime(existing["date"])
        top_daily["existing_rule_based"] = existing[["date", "strategy_return"]].copy()

    yearly, quarterly = build_yearly_comparison(top_daily, returns, dates)
    yearly.to_csv(YEARLY_COMPARISON_PATH, index=False)
    quarterly.to_csv(QUARTERLY_COMPARISON_PATH, index=False)

    weak_summary, weak_search = build_weak_theme_diagnostics(scores, returns)
    weak_summary.to_csv(WEAK_CONDITION_PATH, index=False)
    weak_search.to_csv(WEAK_SEARCH_PATH, index=False)
    weak_strategies = run_weak_strategy_suite(scores, dates, panel_by_date, returns)
    weak_strategies.to_csv(WEAK_STRATEGY_PATH, index=False)

    universe_audit, overlaps, universe_summary = build_universe_audit(scores)
    universe_audit.to_csv(UNIVERSE_AUDIT_PATH, index=False)
    overlaps.to_csv(OVERLAP_PATH, index=False)

    oracle = results[results["is_oracle"]].sort_values("cagr", ascending=False).head(10)
    summary = {
        "strategy_sweep_count": int(len(results)),
        "best_non_oracle_by_cagr": best_cagr_cfg,
        "best_non_oracle_by_sharpe": best_sharpe_cfg,
        "top_non_oracle": results[~results["is_oracle"]].head(20).to_dict(orient="records"),
        "top_oracle": oracle.to_dict(orient="records"),
        "weak_conditions": weak_summary.to_dict(orient="records"),
        "best_weak_condition_search_for_shorts": weak_search.head(20).to_dict(orient="records") if not weak_search.empty else [],
        "weak_strategy_results": weak_strategies.to_dict(orient="records"),
        "universe_summary": universe_summary,
        "output_files": {
            "strategy_sweep": str(STRATEGY_SWEEP_PATH),
            "weak_condition_summary": str(WEAK_CONDITION_PATH),
            "weak_condition_search": str(WEAK_SEARCH_PATH),
            "weak_strategy_results": str(WEAK_STRATEGY_PATH),
            "yearly_comparison": str(YEARLY_COMPARISON_PATH),
            "quarterly_2025_comparison": str(QUARTERLY_COMPARISON_PATH),
            "universe_audit": str(UNIVERSE_AUDIT_PATH),
            "overlaps": str(OVERLAP_PATH),
            "frontier_plot": str(FRONTIER_PLOT_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\ntop non-oracle strategy results")
    display_cols = [
        "label",
        "basket",
        "rank_col",
        "top_n",
        "exit_rank",
        "min_hold",
        "exposure_mode",
        "cagr",
        "sharpe",
        "max_drawdown",
        "turnover",
    ]
    print(results[~results["is_oracle"]][display_cols].head(12).to_string(index=False))
    print("\nweak theme condition summary")
    print(weak_summary[["condition", "observations", "avg_fwd_5d_theme_return", "pct_negative_fwd_5d", "short_ev_5d_before_cost"]].to_string(index=False))
    print(f"\nsaved deep exploration outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
