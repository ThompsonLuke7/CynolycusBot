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

OUT_DIR = OUTPUT_DIR / "regime_tests"


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


def pick_theme_leaders(leaders: pd.DataFrame, date: pd.Timestamp, theme: str, leaders_per_theme: int) -> list[str]:
    day = leaders[(leaders["date"].eq(date)) & (leaders["theme"].eq(theme))]
    return day.sort_values("leader_rank").head(leaders_per_theme)["ticker"].dropna().tolist()


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
        "hit_rate": float((returns > 0).mean()),
        "turnover": float(turnover.mean()),
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else None,
    }


def should_enter(row: pd.Series, mode: str) -> bool:
    if mode == "regime":
        return row["theme_regime_rank"] <= 5
    if mode == "heat_and_regime":
        return row["theme_heat_rank"] <= 5 and row["theme_regime_rank"] <= 10
    raise ValueError(f"unknown mode {mode}")


def run_regime_test(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    theme_daily: pd.DataFrame,
    *,
    mode: str,
    exit_rank: int,
    leaders_per_theme: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(set(scores["date"]).intersection(returns.index))
    score_lookup = scores.set_index(["date", "theme"])
    active: dict[str, dict] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []
    trades = []
    periods = []

    for i, date in enumerate(dates):
        if active:
            holdings = []
            for theme, state in active.items():
                for ticker in state["tickers"]:
                    holdings.append({"ticker": ticker, "weight": state["weight"] / len(state["tickers"])})
            weights = pd.DataFrame(holdings).groupby("ticker")["weight"].sum()
            weights = weights / weights.abs().sum()
            gross_return = float((returns.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())
        else:
            weights = pd.Series(dtype=float)
            gross_return = 0.0

        updated = dict(active)
        signal_date = date
        for theme, state in list(active.items()):
            row = score_lookup.loc[(signal_date, theme)] if (signal_date, theme) in score_lookup.index else pd.Series(dtype=float)
            rank = row.get("theme_regime_rank", np.nan)
            if pd.isna(rank) or rank > exit_rank:
                periods.append({"theme": theme, "start": state["start"], "end": signal_date, "days": int(max(1, i - state["start_i"] + 1))})
                updated.pop(theme, None)
                trades.append({"date": signal_date, "action": "exit", "mode": mode, "theme": theme, "theme_regime_rank": rank})

        day = scores[scores["date"].eq(signal_date)].sort_values("theme_regime_rank")
        for entry in day.itertuples(index=False):
            row = pd.Series(entry._asdict())
            if entry.theme in updated or not should_enter(row, mode):
                continue
            tickers = pick_theme_leaders(leaders, signal_date, entry.theme, leaders_per_theme)
            if not tickers:
                continue
            updated[entry.theme] = {"start": signal_date, "start_i": i + 1, "tickers": tickers, "weight": 1.0}
            trades.append(
                {
                    "date": signal_date,
                    "action": "enter",
                    "mode": mode,
                    "theme": entry.theme,
                    "theme_heat_rank": entry.theme_heat_rank,
                    "theme_regime_rank": entry.theme_regime_rank,
                    "theme_regime_score": entry.theme_regime_score,
                    "tickers": "|".join(tickers),
                }
            )

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
        rows.append(
            {
                "date": date,
                "mode": mode,
                "strategy_return": gross_return - cost,
                "gross_return": gross_return,
                "turnover": turnover,
                "n_active_themes": len(active),
                "active_themes": "|".join(sorted(active)),
            }
        )

    if dates:
        last_date = dates[-1]
        last_i = len(dates) - 1
        for theme, state in active.items():
            periods.append({"theme": theme, "start": state["start"], "end": last_date, "days": int(max(1, last_i - state["start_i"] + 1))})

    return pd.DataFrame(rows), pd.DataFrame(trades), enrich_duration(pd.DataFrame(periods), scores, theme_daily)


def enrich_duration(periods: pd.DataFrame, scores: pd.DataFrame, theme_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in periods.itertuples(index=False):
        score_slice = scores[(scores["theme"].eq(row.theme)) & (scores["date"].between(row.start, row.end))]
        ret_slice = theme_daily[(theme_daily["theme"].eq(row.theme)) & (theme_daily["date"].between(row.start, row.end))]
        rows.append(
            {
                "theme": row.theme,
                "start": row.start,
                "end": row.end,
                "days": row.days,
                "avg_rank": float(score_slice["theme_regime_rank"].mean()) if not score_slice.empty else np.nan,
                "max_rank": float(score_slice["theme_regime_rank"].max()) if not score_slice.empty else np.nan,
                "total_return": float((1.0 + ret_slice["theme_return_1d"].fillna(0.0)).prod() - 1.0) if not ret_slice.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["start", "theme"]) if rows else pd.DataFrame()


def duration_stats(duration: pd.DataFrame) -> dict[str, float]:
    if duration.empty:
        return {"mean_theme_hold_days": 0.0, "median_hold_days": 0.0, "p75_hold_days": 0.0}
    return {
        "mean_theme_hold_days": float(duration["days"].mean()),
        "median_hold_days": float(duration["days"].median()),
        "p75_hold_days": float(duration["days"].quantile(0.75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test slow theme regime entry/exit rules.")
    parser.add_argument("--exit-rank", type=int, default=12)
    parser.add_argument("--leaders-per-theme", type=int, default=LEADERS_PER_THEME)
    args = parser.parse_args()

    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    theme_daily = pd.read_parquet(THEME_DAILY_PATH)
    for frame in (scores, leaders, theme_daily):
        frame["date"] = pd.to_datetime(frame["date"])
    scores = tradable_scores(scores)
    returns = load_stock_returns()

    metrics = {}
    durations = []
    for mode in ["regime", "heat_and_regime"]:
        bt, trades, duration = run_regime_test(
            scores,
            leaders,
            returns,
            theme_daily,
            mode=mode,
            exit_rank=args.exit_rank,
            leaders_per_theme=args.leaders_per_theme,
        )
        bt.to_csv(OUT_DIR / f"{mode}_daily.csv", index=False)
        trades.to_csv(OUT_DIR / f"{mode}_trades.csv", index=False)
        duration.to_csv(OUT_DIR / f"{mode}_theme_duration_report.csv", index=False)
        metrics[mode] = {**perf_stats(bt["strategy_return"], bt["turnover"]), **duration_stats(duration)}
        duration = duration.copy()
        duration["mode"] = mode
        durations.append(duration)

    all_duration = pd.concat(durations, ignore_index=True) if durations else pd.DataFrame()
    top_duration = (
        all_duration.groupby(["mode", "theme"])["days"]
        .mean()
        .reset_index(name="avg_regime_duration")
        .sort_values(["mode", "avg_regime_duration"], ascending=[True, False])
    )
    top_duration.to_csv(OUT_DIR / "top_themes_by_average_regime_duration.csv", index=False)
    (OUT_DIR / "regime_backtest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved regime outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
