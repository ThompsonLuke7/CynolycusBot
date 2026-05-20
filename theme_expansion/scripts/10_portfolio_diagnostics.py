from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    BACKTEST_DIR,
    DAILY_BARS_PATH,
    HOLD_DAYS,
    LEADERS_PER_THEME,
    OUTPUT_DIR,
    REBALANCE_EVERY_DAYS,
    THEME_DAILY_PATH,
    THEME_LEADERS_PATH,
    THEME_SCORES_PATH,
    TOP_N_THEMES,
    TRANSACTION_COST_BPS,
    ensure_dirs,
    load_theme_memberships,
)

DIAG_DIR = OUTPUT_DIR / "diagnostics"


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
    return out.dropna(subset=["theme_rank"]).sort_values(["date", "theme_rank"])


def top_themes_for(scores: pd.DataFrame, date: pd.Timestamp, top_n: int) -> list[str]:
    return scores[scores["date"].eq(date)].nsmallest(top_n, "theme_rank")["theme"].tolist()


def pick_daily_tickers(leaders: pd.DataFrame, date: pd.Timestamp, themes: list[str], leaders_per_theme: int) -> pd.DataFrame:
    day = leaders[(leaders["date"].eq(date)) & (leaders["theme"].isin(themes))]
    return day[day["leader_rank"] <= leaders_per_theme].copy()


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


def simulate_exit_mode(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    exit_mode: str,
    top_n: int,
    leaders_per_theme: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(set(scores["date"]).intersection(returns.index))
    score_lookup = scores.set_index(["date", "theme"])
    positions: list[dict] = []
    prev_weights = pd.Series(dtype=float)
    rows = []
    overlap_rows = []
    trade_rows = []

    for i, date in enumerate(dates):
        signal_date = dates[i - 1] if i > 0 else None
        kept_positions = []
        for pos in positions:
            if pos["exit_i"] < i:
                continue
            if signal_date is None:
                kept_positions.append(pos)
                continue
            if exit_mode == "rank_exit":
                rank = (
                    score_lookup.loc[(signal_date, pos["theme"]), "theme_rank"]
                    if (signal_date, pos["theme"]) in score_lookup.index
                    else np.nan
                )
                if pd.notna(rank) and rank > 5:
                    continue
            elif exit_mode == "decay_exit":
                score = (
                    score_lookup.loc[(signal_date, pos["theme"]), "theme_score"]
                    if (signal_date, pos["theme"]) in score_lookup.index
                    else np.nan
                )
                rolling = (
                    score_lookup.loc[(signal_date, pos["theme"]), "theme_score_rolling_3d"]
                    if (signal_date, pos["theme"]) in score_lookup.index
                    else np.nan
                )
                if pd.notna(score) and pd.notna(rolling) and score < rolling:
                    continue
            kept_positions.append(pos)
        positions = kept_positions

        if positions:
            pos_frame = pd.DataFrame(positions)
            weights = pos_frame.groupby("ticker")["weight"].sum()
            weights = weights / weights.abs().sum()
            gross_return = float((returns.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())
            lot_counts = pos_frame.groupby("ticker").size()
        else:
            weights = pd.Series(dtype=float)
            gross_return = 0.0
            lot_counts = pd.Series(dtype=float)

        turnover = 0.0
        selected = pd.DataFrame()
        top_themes = []
        if i % REBALANCE_EVERY_DAYS == 0:
            top_themes = top_themes_for(scores, date, top_n)
            selected = pick_daily_tickers(leaders, date, top_themes, leaders_per_theme)
            tickers = sorted(selected["ticker"].dropna().unique())
            existing = set(weights.index)
            overlap = len(existing & set(tickers)) / len(tickers) if tickers else 0.0
            if tickers:
                new_lot = pd.Series(1.0 / len(tickers), index=tickers)
                after_trade = weights.add(new_lot, fill_value=0.0)
                after_trade = after_trade / after_trade.abs().sum()
                turnover = float(after_trade.sub(prev_weights, fill_value=0.0).abs().sum())
                prev_weights = after_trade
                for ticker in tickers:
                    theme = selected[selected["ticker"].eq(ticker)].sort_values("leader_rank")["theme"].iloc[0]
                    positions.append({"ticker": ticker, "theme": theme, "weight": 1.0 / len(tickers), "entry_i": i + 1, "exit_i": i + HOLD_DAYS})
                    trade_rows.append({"signal_date": date, "theme": theme, "ticker": ticker, "exit_mode": exit_mode})
            overlap_rows.append(
                {
                    "date": date,
                    "exit_mode": exit_mode,
                    "top_themes": "|".join(top_themes),
                    "selected_tickers": "|".join(tickers),
                    "selected_overlap_existing_pct": overlap,
                    "max_lots_per_ticker": int(lot_counts.max()) if len(lot_counts) else 0,
                    "avg_lots_per_ticker": float(lot_counts.mean()) if len(lot_counts) else 0.0,
                    "n_active_lots": int(len(positions)),
                    "n_active_tickers": int(len(weights)),
                }
            )

        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append({"date": date, "exit_mode": exit_mode, "strategy_return": gross_return - cost, "turnover": turnover})

    return pd.DataFrame(rows), pd.DataFrame(overlap_rows), pd.DataFrame(trade_rows)


def theme_duration(scores: pd.DataFrame, theme_daily: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top = scores[scores["theme_rank"] <= top_n][["date", "theme"]].copy()
    ret = theme_daily[["date", "theme", "theme_return_1d"]].copy()
    rows = []
    for theme, frame in top.groupby("theme"):
        dates = sorted(frame["date"].tolist())
        if not dates:
            continue
        start = dates[0]
        prev = dates[0]
        for date in dates[1:]:
            if (date - prev).days > 4:
                segment = ret[(ret["theme"].eq(theme)) & (ret["date"].between(start, prev))]
                rows.append(
                    {
                        "theme": theme,
                        "start": start,
                        "end": prev,
                        "days": int(len(segment)),
                        "peak_return": float(segment["theme_return_1d"].max()),
                        "avg_return": float(segment["theme_return_1d"].mean()),
                    }
                )
                start = date
            prev = date
        segment = ret[(ret["theme"].eq(theme)) & (ret["date"].between(start, prev))]
        rows.append(
            {
                "theme": theme,
                "start": start,
                "end": prev,
                "days": int(len(segment)),
                "peak_return": float(segment["theme_return_1d"].max()),
                "avg_return": float(segment["theme_return_1d"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["start", "theme"])
    return out


def overlap_reports(theme_daily: pd.DataFrame) -> None:
    memberships = load_theme_memberships()
    memberships = memberships[memberships["is_tradable"].fillna(False).astype(bool)]
    themes = sorted(memberships["theme"].unique())
    rows = []
    for left in themes:
        left_tickers = set(memberships[memberships["theme"].eq(left)]["ticker"])
        for right in themes:
            right_tickers = set(memberships[memberships["theme"].eq(right)]["ticker"])
            union = left_tickers | right_tickers
            rows.append({"theme": left, "other_theme": right, "jaccard_overlap": len(left_tickers & right_tickers) / len(union) if union else 0.0})
    pd.DataFrame(rows).pivot(index="theme", columns="other_theme", values="jaccard_overlap").to_csv(DIAG_DIR / "theme_overlap_matrix.csv")

    ret = theme_daily.pivot(index="date", columns="theme", values="theme_return_1d").sort_index()
    corr = ret.corr()
    corr.to_csv(DIAG_DIR / "theme_return_corr_matrix.csv")
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and value > 0.70:
                pairs.append({"theme": left, "other_theme": right, "return_corr": float(value)})
    pd.DataFrame(pairs).sort_values("return_corr", ascending=False).to_csv(DIAG_DIR / "high_corr_theme_pairs.csv", index=False)


def trade_explanations(trades: pd.DataFrame, scores: pd.DataFrame, theme_duration_df: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    future = curve.shift(-HOLD_DAYS) / curve - 1.0
    duration_lookup = theme_duration_df.groupby("theme")["days"].median().rename("theme_duration_median").reset_index()
    score_cols = ["date", "theme", "theme_score", "theme_rank", "theme_return_5d", "theme_vs_spy_5d", "theme_breadth", "entropy_score"]
    out = trades.merge(scores[score_cols], left_on=["signal_date", "theme"], right_on=["date", "theme"], how="left")
    out = out.merge(duration_lookup, on="theme", how="left")
    out["trade_return_5d"] = [
        future.at[row.signal_date, row.ticker] if row.signal_date in future.index and row.ticker in future.columns else np.nan
        for row in out.itertuples(index=False)
    ]
    leaders = out.groupby(["signal_date", "theme"])["ticker"].apply(lambda s: "|".join(s.head(5))).rename("leaders").reset_index()
    out = out.merge(leaders, on=["signal_date", "theme"], how="left")
    return out.drop(columns=["date"], errors="ignore").sort_values(["signal_date", "theme", "ticker"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose theme persistence, overlap, exits, and position overlap.")
    parser.add_argument("--top-themes", type=int, default=TOP_N_THEMES)
    parser.add_argument("--leaders-per-theme", type=int, default=LEADERS_PER_THEME)
    args = parser.parse_args()

    ensure_dirs()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    theme_daily = pd.read_parquet(THEME_DAILY_PATH)
    for frame in (scores, leaders, theme_daily):
        frame["date"] = pd.to_datetime(frame["date"])
    scores = tradable_scores(scores)
    scores["theme_score_rolling_3d"] = scores.sort_values(["theme", "date"]).groupby("theme")["theme_score"].transform(
        lambda s: s.rolling(3, min_periods=2).mean()
    )
    returns = load_stock_returns()

    duration = theme_duration(scores, theme_daily, args.top_themes)
    duration.to_csv(DIAG_DIR / "theme_duration.csv", index=False)
    overlap_reports(theme_daily)

    metrics = []
    all_overlap = []
    all_trades = []
    for mode in ["fixed_5d", "rank_exit", "decay_exit"]:
        bt, overlap, trades = simulate_exit_mode(
            scores,
            leaders,
            returns,
            exit_mode=mode,
            top_n=args.top_themes,
            leaders_per_theme=args.leaders_per_theme,
        )
        bt.to_csv(DIAG_DIR / f"exit_study_{mode}_daily.csv", index=False)
        metrics.append({"exit_mode": mode, **perf_stats(bt["strategy_return"], bt["turnover"])})
        all_overlap.append(overlap)
        all_trades.append(trades)

    pd.DataFrame(metrics).to_csv(DIAG_DIR / "exit_study_metrics.csv", index=False)
    overlap = pd.concat(all_overlap, ignore_index=True)
    overlap.to_csv(DIAG_DIR / "position_overlap.csv", index=False)
    trades = pd.concat(all_trades, ignore_index=True)
    explanations = trade_explanations(trades, scores, duration, returns)
    explanations.to_csv(DIAG_DIR / "trade_explanations.csv", index=False)

    summary = {
        "avg_top_theme_duration_days": float(duration["days"].mean()) if not duration.empty else 0.0,
        "median_top_theme_duration_days": float(duration["days"].median()) if not duration.empty else 0.0,
        "avg_selected_overlap_existing_pct": float(overlap["selected_overlap_existing_pct"].mean()) if not overlap.empty else 0.0,
        "avg_active_lots": float(overlap["n_active_lots"].mean()) if not overlap.empty else 0.0,
        "avg_active_tickers": float(overlap["n_active_tickers"].mean()) if not overlap.empty else 0.0,
    }
    (DIAG_DIR / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved diagnostics -> {DIAG_DIR}")


if __name__ == "__main__":
    main()
