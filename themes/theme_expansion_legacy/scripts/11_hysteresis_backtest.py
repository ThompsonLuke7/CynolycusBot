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
    TOP_N_THEMES,
    TRANSACTION_COST_BPS,
    ensure_dirs,
)

OUT_DIR = OUTPUT_DIR / "hysteresis"


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
    out = out.dropna(subset=["theme_rank"]).sort_values(["theme", "date"])
    out["theme_score_rolling_3d"] = out.groupby("theme")["theme_score"].transform(
        lambda s: s.rolling(3, min_periods=2).mean()
    )
    return out.sort_values(["date", "theme_rank"])


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
        "sortino": float(np.sqrt(252.0) * returns.mean() / losers.std()) if losers.std() else 0.0,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "turnover": float(turnover.mean()),
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else None,
    }


def should_exit(row: pd.Series, exit_rank: int, use_decay_exit: bool) -> bool:
    if pd.isna(row.get("theme_rank")):
        return True
    if row["theme_rank"] > exit_rank:
        return True
    if use_decay_exit:
        rolling = row.get("theme_score_rolling_3d")
        if pd.notna(rolling) and row["theme_score"] < rolling:
            return True
    return False


def run_hysteresis(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    enter_rank: int,
    exit_rank: int,
    leaders_per_theme: int,
    use_decay_exit: bool,
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
                    holdings.append({"theme": theme, "ticker": ticker, "weight": state["weight"] / len(state["tickers"])})
            hold_frame = pd.DataFrame(holdings)
            weights = hold_frame.groupby("ticker")["weight"].sum()
            weights = weights / weights.abs().sum()
            gross_return = float((returns.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())
        else:
            weights = pd.Series(dtype=float)
            gross_return = 0.0

        signal_date = date
        updated = dict(active)
        for theme, state in list(active.items()):
            if (signal_date, theme) not in score_lookup.index:
                exit_now = True
                row = pd.Series(dtype=float)
            else:
                row = score_lookup.loc[(signal_date, theme)]
                exit_now = should_exit(row, exit_rank, use_decay_exit)
            if exit_now:
                periods.append(
                    {
                        "theme": theme,
                        "start": state["start"],
                        "end": signal_date,
                        "days": int(max(1, i - state["start_i"] + 1)),
                    }
                )
                updated.pop(theme, None)
                trades.append(
                    {
                        "date": signal_date,
                        "action": "exit",
                        "theme": theme,
                        "theme_rank": row.get("theme_rank", np.nan),
                        "theme_score": row.get("theme_score", np.nan),
                        "tickers": "|".join(state["tickers"]),
                    }
                )

        day = scores[scores["date"].eq(signal_date)]
        entries = day[day["theme_rank"] <= enter_rank].sort_values("theme_rank")
        for entry in entries.itertuples(index=False):
            if entry.theme in updated:
                continue
            tickers = pick_theme_leaders(leaders, signal_date, entry.theme, leaders_per_theme)
            if not tickers:
                continue
            updated[entry.theme] = {
                "start": signal_date,
                "start_i": i + 1,
                "tickers": tickers,
                "weight": 1.0,
            }
            trades.append(
                {
                    "date": signal_date,
                    "action": "enter",
                    "theme": entry.theme,
                    "theme_rank": entry.theme_rank,
                    "theme_score": entry.theme_score,
                    "tickers": "|".join(tickers),
                }
            )

        if updated:
            theme_weight = 1.0 / len(updated)
            for state in updated.values():
                state["weight"] = theme_weight
            new_holdings = []
            for theme, state in updated.items():
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
                "strategy_return": gross_return - cost,
                "gross_return": gross_return,
                "turnover": turnover,
                "n_active_themes": len(active),
                "n_active_tickers": len(new_weights),
                "active_themes": "|".join(sorted(active)),
            }
        )

    if dates:
        last_date = dates[-1]
        last_i = len(dates) - 1
        for theme, state in active.items():
            periods.append({"theme": theme, "start": state["start"], "end": last_date, "days": int(max(1, last_i - state["start_i"] + 1))})

    return pd.DataFrame(rows), pd.DataFrame(trades), pd.DataFrame(periods)


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
                "avg_rank": float(score_slice["theme_rank"].mean()) if not score_slice.empty else np.nan,
                "max_rank": float(score_slice["theme_rank"].max()) if not score_slice.empty else np.nan,
                "total_return": float((1.0 + ret_slice["theme_return_1d"].fillna(0.0)).prod() - 1.0) if not ret_slice.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["start", "theme"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest theme entry/exit hysteresis without fixed 5d holding.")
    parser.add_argument("--enter-rank", type=int, default=TOP_N_THEMES)
    parser.add_argument("--exit-rank", type=int, default=8)
    parser.add_argument("--leaders-per-theme", type=int, default=LEADERS_PER_THEME)
    parser.add_argument("--rank-only", action="store_true", help="Disable score < rolling_3d_mean decay exit.")
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
    mode_name = "rank_only" if args.rank_only else "rank_decay"

    bt, trades, periods = run_hysteresis(
        scores,
        leaders,
        returns,
        enter_rank=args.enter_rank,
        exit_rank=args.exit_rank,
        leaders_per_theme=args.leaders_per_theme,
        use_decay_exit=not args.rank_only,
    )
    duration = enrich_duration(periods, scores, theme_daily)
    metrics = {
        "strategy": perf_stats(bt["strategy_return"], bt["turnover"]),
        "duration": {
            "mean_duration": float(duration["days"].mean()) if not duration.empty else 0.0,
            "median_duration": float(duration["days"].median()) if not duration.empty else 0.0,
            "p75_duration": float(duration["days"].quantile(0.75)) if not duration.empty else 0.0,
        },
    }
    bt.to_csv(OUT_DIR / f"hysteresis_{mode_name}_daily.csv", index=False)
    trades.to_csv(OUT_DIR / f"hysteresis_{mode_name}_trades.csv", index=False)
    duration.to_csv(OUT_DIR / f"theme_duration_report_{mode_name}.csv", index=False)
    (OUT_DIR / f"hysteresis_metrics_{mode_name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved hysteresis outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
