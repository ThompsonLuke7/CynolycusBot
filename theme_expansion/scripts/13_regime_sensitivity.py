from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DAILY_BARS_PATH,
    LEADERS_PER_THEME,
    OUTPUT_DIR,
    THEME_DAILY_PATH,
    THEME_LEADERS_PATH,
    THEME_SCORES_PATH,
    TRANSACTION_COST_BPS,
    ensure_dirs,
)

OUT_DIR = OUTPUT_DIR / "regime_sensitivity"


def load_stock_returns() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    bars = bars.sort_values(["ticker", "date"])
    bars["stock_return_1d"] = bars.groupby("ticker")["px"].pct_change()
    return bars.pivot(index="date", columns="ticker", values="stock_return_1d").sort_index()


def tradable_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    if "is_tradable" in out.columns:
        out = out[out["is_tradable"].fillna(False).astype(bool)]
    return out.dropna(subset=["theme_regime_rank"]).sort_values(["date", "theme_regime_rank"])


def pick_theme_leaders(leaders: pd.DataFrame, date: pd.Timestamp, theme: str) -> list[str]:
    day = leaders[(leaders["date"].eq(date)) & (leaders["theme"].eq(theme))]
    return day.sort_values("leader_rank").head(LEADERS_PER_THEME)["ticker"].dropna().tolist()


def perf_stats(returns: pd.Series, turnover: pd.Series) -> dict[str, float | None]:
    equity = (1.0 + returns).cumprod()
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    loss_sum = abs(losers.sum())
    return {
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "sharpe": float(np.sqrt(252.0) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.mean()),
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else None,
    }


def run_case(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    enter_rank: int,
    exit_rank: int,
    min_hold_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(set(scores["date"]).intersection(returns.index))
    score_lookup = scores.set_index(["date", "theme"])
    active: dict[str, dict] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []
    periods = []

    for i, date in enumerate(dates):
        if active:
            holdings = []
            for state in active.values():
                for ticker in state["tickers"]:
                    holdings.append({"ticker": ticker, "weight": state["weight"] / len(state["tickers"])})
            weights = pd.DataFrame(holdings).groupby("ticker")["weight"].sum()
            weights = weights / weights.abs().sum()
            gross_return = float((returns.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())
        else:
            gross_return = 0.0

        updated = dict(active)
        for theme, state in list(active.items()):
            held_days = i - state["start_i"] + 1
            row = score_lookup.loc[(date, theme)] if (date, theme) in score_lookup.index else pd.Series(dtype=float)
            rank = row.get("theme_regime_rank", np.nan)
            if held_days >= min_hold_days and (pd.isna(rank) or rank > exit_rank):
                periods.append({"theme": theme, "start": state["start"], "end": date, "days": int(max(1, held_days))})
                updated.pop(theme, None)

        day = scores[scores["date"].eq(date)].sort_values("theme_regime_rank")
        entries = day[day["theme_regime_rank"] <= enter_rank]
        for entry in entries.itertuples(index=False):
            if entry.theme in updated:
                continue
            tickers = pick_theme_leaders(leaders, date, entry.theme)
            if not tickers:
                continue
            updated[entry.theme] = {"start": date, "start_i": i + 1, "tickers": tickers, "weight": 1.0}

        if updated:
            theme_weight = 1.0 / len(updated)
            for state in updated.values():
                state["weight"] = theme_weight
            new_holdings = []
            for state in updated.values():
                for ticker in state["tickers"]:
                    new_holdings.append({"ticker": ticker, "weight": state["weight"] / len(state["tickers"])})
            new_weights = pd.DataFrame(new_holdings).groupby("ticker")["weight"].sum()
            new_weights = new_weights / new_weights.abs().sum()
        else:
            new_weights = pd.Series(dtype=float)
        turnover = float(new_weights.sub(prev_weights, fill_value=0.0).abs().sum())
        prev_weights = new_weights
        active = updated
        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append({"date": date, "strategy_return": gross_return - cost, "turnover": turnover, "n_active_themes": len(active)})

    if dates:
        last_date = dates[-1]
        last_i = len(dates) - 1
        for theme, state in active.items():
            periods.append({"theme": theme, "start": state["start"], "end": last_date, "days": int(max(1, last_i - state["start_i"] + 1))})

    return pd.DataFrame(rows), pd.DataFrame(periods)


def summarize_duration(periods: pd.DataFrame) -> dict[str, float]:
    if periods.empty:
        return {"mean_hold_days": 0.0, "median_hold_days": 0.0}
    return {
        "mean_hold_days": float(periods["days"].mean()),
        "median_hold_days": float(periods["days"].median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity tests on slow regime entry/exit logic.")
    args = parser.parse_args()
    _ = args

    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    for frame in (scores, leaders):
        frame["date"] = pd.to_datetime(frame["date"])
    scores = tradable_scores(scores)
    returns = load_stock_returns()

    rows = []
    best_periods = None
    best_key = None
    best_sharpe = -np.inf
    for enter_rank in [3, 5, 8]:
        for exit_rank in [10, 12, 15]:
            for min_hold_days in [0, 3, 5]:
                bt, periods = run_case(
                    scores,
                    leaders,
                    returns,
                    enter_rank=enter_rank,
                    exit_rank=exit_rank,
                    min_hold_days=min_hold_days,
                )
                metrics = {
                    "enter_rank": enter_rank,
                    "exit_rank": exit_rank,
                    "min_hold_days": min_hold_days,
                    **perf_stats(bt["strategy_return"], bt["turnover"]),
                    **summarize_duration(periods),
                }
                rows.append(metrics)
                if pd.notna(metrics["sharpe"]) and metrics["sharpe"] > best_sharpe:
                    best_sharpe = float(metrics["sharpe"])
                    best_periods = periods.copy()
                    best_key = (enter_rank, exit_rank, min_hold_days)

    out = pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False)
    out.to_csv(OUT_DIR / "regime_sensitivity_results.csv", index=False)
    if best_periods is not None and best_key is not None:
        best_periods["enter_rank"] = best_key[0]
        best_periods["exit_rank"] = best_key[1]
        best_periods["min_hold_days"] = best_key[2]
        best_periods.to_csv(OUT_DIR / "best_regime_sensitivity_periods.csv", index=False)
    (OUT_DIR / "regime_sensitivity_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"saved sensitivity outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
