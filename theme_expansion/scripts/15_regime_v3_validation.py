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
    THEME_LEADERS_PATH,
    THEME_SCORES_PATH,
    TRANSACTION_COST_BPS,
    ensure_dirs,
)

OUT_DIR = OUTPUT_DIR / "regime_v3_validation"
ENTER_RANK = 3
EXIT_RANK = 12
MIN_HOLD_DAYS = 5
ROLLING_WINDOW = 252


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
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    loss_sum = abs(losers.sum())
    return {
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "sharpe": float(np.sqrt(252.0) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "max_dd": float(drawdown.min()),
        "turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else None,
    }


def simulate(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    exclude_theme: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if exclude_theme:
        scores = scores[~scores["theme"].eq(exclude_theme)].copy()
        leaders = leaders[~leaders["theme"].eq(exclude_theme)].copy()

    dates = sorted(set(scores["date"]).intersection(returns.index))
    score_lookup = scores.set_index(["date", "theme"])
    active: dict[str, dict] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []
    periods = []
    contrib_rows = []

    for i, date in enumerate(dates):
        gross = 0.0
        if active:
            for theme, state in active.items():
                tickers = [t for t in state["tickers"] if t in returns.columns]
                if not tickers:
                    continue
                ticker_weight = state["weight"] / len(tickers)
                theme_ret = float((returns.loc[date].reindex(tickers).fillna(0.0) * ticker_weight).sum())
                gross += theme_ret
                contrib_rows.append(
                    {
                        "date": date,
                        "theme": theme,
                        "daily_contribution": theme_ret,
                        "theme_weight": state["weight"],
                        "tickers": "|".join(tickers),
                    }
                )

        updated = dict(active)
        for theme, state in list(active.items()):
            held_days = i - state["start_i"] + 1
            row = score_lookup.loc[(date, theme)] if (date, theme) in score_lookup.index else pd.Series(dtype=float)
            rank = row.get("theme_regime_rank", np.nan)
            if held_days >= MIN_HOLD_DAYS and (pd.isna(rank) or rank > EXIT_RANK):
                periods.append({"theme": theme, "start": state["start"], "end": date, "days": int(max(1, held_days))})
                updated.pop(theme, None)

        day = scores[scores["date"].eq(date)].sort_values("theme_regime_rank")
        entries = day[day["theme_regime_rank"] <= ENTER_RANK]
        for entry in entries.itertuples(index=False):
            if entry.theme in updated:
                continue
            tickers = [t for t in pick_theme_leaders(leaders, date, entry.theme) if t in returns.columns]
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
        rows.append(
            {
                "date": date,
                "strategy_return": gross - cost,
                "gross_return": gross,
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

    return pd.DataFrame(rows), pd.DataFrame(periods), pd.DataFrame(contrib_rows)


def period_report(bt: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("2020_2021", "2020-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", "2026-12-31"),
    ]
    rows = []
    for label, start, end in specs:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        sub = bt[(bt["date"] >= start_ts) & (bt["date"] <= end_ts)]
        if sub.empty:
            continue
        sub_periods = periods[(periods["start"] >= start_ts) & (periods["start"] <= end_ts)]
        stats = perf_stats(sub["strategy_return"], sub["turnover"])
        rows.append(
            {
                "period": label,
                **stats,
                "avg_hold": float(sub_periods["days"].mean()) if not sub_periods.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rolling_report(bt: pd.DataFrame) -> pd.DataFrame:
    frame = bt[["date", "strategy_return"]].copy().sort_values("date")
    rows = []
    returns = frame["strategy_return"].fillna(0.0)
    for end_idx in range(ROLLING_WINDOW - 1, len(frame)):
        window = returns.iloc[end_idx - ROLLING_WINDOW + 1 : end_idx + 1]
        equity = (1.0 + window).cumprod()
        dd = equity / equity.cummax() - 1.0
        cagr = equity.iloc[-1] - 1.0
        sharpe = np.sqrt(252.0) * window.mean() / window.std() if window.std() else 0.0
        rows.append(
            {
                "date": frame["date"].iloc[end_idx],
                "rolling_12m_sharpe": float(sharpe),
                "rolling_12m_cagr": float(cagr),
                "rolling_12m_dd": float(dd.min()),
            }
        )
    return pd.DataFrame(rows)


def leave_one_theme_out(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    themes: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    selected_themes = themes or sorted(scores["theme"].unique())
    for theme in selected_themes:
        bt, periods, _ = simulate(scores, leaders, returns, exclude_theme=theme)
        stats = perf_stats(bt["strategy_return"], bt["turnover"])
        rows.append(
            {
                "excluded_theme": theme,
                **stats,
                "mean_hold_days": float(periods["days"].mean()) if not periods.empty else np.nan,
                "median_hold_days": float(periods["days"].median()) if not periods.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def theme_contribution(contrib: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    if contrib.empty:
        return pd.DataFrame()
    grouped = contrib.groupby("theme").agg(
        total_pnl=("daily_contribution", "sum"),
        avg_return=("daily_contribution", "mean"),
        active_days=("date", "count"),
    )
    trade_count = periods.groupby("theme").size().rename("trade_count")
    avg_hold = periods.groupby("theme")["days"].mean().rename("avg_hold")
    out = grouped.join(trade_count, how="left").join(avg_hold, how="left").fillna({"trade_count": 0, "avg_hold": 0.0})
    return out.reset_index().sort_values("total_pnl", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen regime v3 robustness.")
    parser.add_argument("--skip-leave-one", action="store_true", help="Skip expensive leave-one-theme-out attribution.")
    parser.add_argument("--leave-one-start", type=int, default=None, help="Start index for chunked leave-one-theme-out.")
    parser.add_argument("--leave-one-end", type=int, default=None, help="End index for chunked leave-one-theme-out.")
    args = parser.parse_args()

    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    for frame in (scores, leaders):
        frame["date"] = pd.to_datetime(frame["date"])
    scores = tradable_scores(scores)
    returns = load_stock_returns()

    bt, periods, contrib = simulate(scores, leaders, returns)
    bt.to_csv(OUT_DIR / "regime_v3_daily.csv", index=False)
    periods.to_csv(OUT_DIR / "regime_v3_periods.csv", index=False)
    period_report(bt, periods).to_csv(OUT_DIR / "period_performance.csv", index=False)
    rolling_report(bt).to_csv(OUT_DIR / "rolling_12m_metrics.csv", index=False)
    theme_contribution(contrib, periods).to_csv(OUT_DIR / "theme_contribution_report.csv", index=False)
    if not args.skip_leave_one:
        themes = sorted(scores["theme"].unique())
        start = args.leave_one_start
        end = args.leave_one_end
        if start is not None or end is not None:
            start = 0 if start is None else start
            end = len(themes) if end is None else end
            selected = themes[start:end]
            out_path = OUT_DIR / f"leave_one_theme_out_{start:03d}_{end:03d}.csv"
            leave_one_theme_out(scores, leaders, returns, themes=selected).to_csv(out_path, index=False)
        else:
            leave_one_theme_out(scores, leaders, returns).to_csv(OUT_DIR / "leave_one_theme_out.csv", index=False)

    summary = {
        "overall": perf_stats(bt["strategy_return"], bt["turnover"]),
        "mean_hold_days": float(periods["days"].mean()) if not periods.empty else 0.0,
        "median_hold_days": float(periods["days"].median()) if not periods.empty else 0.0,
    }
    (OUT_DIR / "regime_v3_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved validation outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
