from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    BACKTEST_DIR,
    BENCHMARK_TICKERS,
    DAILY_BARS_PATH,
    HOLD_DAYS,
    LEADERS_PER_THEME,
    REBALANCE_EVERY_DAYS,
    THEME_LEADERS_PATH,
    THEME_SCORES_PATH,
    TOP_N_THEMES,
    TRANSACTION_COST_BPS,
    ensure_dirs,
)


def load_stock_returns() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    bars = bars.sort_values(["ticker", "date"])
    bars["stock_return_1d"] = bars.groupby("ticker")["px"].pct_change()
    return bars.pivot(index="date", columns="ticker", values="stock_return_1d").sort_index()


def load_price_panel() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    return bars.pivot(index="date", columns="ticker", values="px").sort_index()


def download_vix(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download(
        "^VIX",
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=["date", "vix"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw.reset_index()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    px_col = "adj_close" if "adj_close" in out.columns else "close"
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out[["date", px_col]].rename(columns={px_col: "vix"})


def build_market_regime() -> pd.DataFrame:
    wide = load_price_panel()
    out = pd.DataFrame(index=wide.index)
    for ticker in ["SPY", "QQQ"]:
        out[f"{ticker.lower()}_above_200dma"] = wide[ticker] > wide[ticker].rolling(200, min_periods=100).mean()
    stock_cols = [
        col
        for col in wide.columns
        if col not in {"SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "SMH", "XLK", "XLF", "XLV", "XLE"}
    ]
    sma200 = wide[stock_cols].rolling(200, min_periods=100).mean()
    out["market_breadth"] = (wide[stock_cols] > sma200).mean(axis=1)
    vix = download_vix(wide.index.min(), wide.index.max())
    if not vix.empty:
        out = out.reset_index().rename(columns={"index": "date"}).merge(vix, on="date", how="left").set_index("date")
        out["vix"] = out["vix"].ffill()
        out["vix_20d_trend"] = out["vix"].pct_change(20)
    else:
        out["vix"] = np.nan
        out["vix_20d_trend"] = np.nan

    risk_off = (
        (~out["spy_above_200dma"].fillna(False) & ~out["qqq_above_200dma"].fillna(False))
        | (out["market_breadth"] < 0.40)
        | (out["vix_20d_trend"] > 0.15)
    )
    risk_on = (
        out["spy_above_200dma"].fillna(False)
        & out["qqq_above_200dma"].fillna(False)
        & (out["market_breadth"] > 0.55)
        & (out["vix_20d_trend"].fillna(0.0) <= 0.10)
    )
    out["regime"] = np.select([risk_on, risk_off], ["risk_on", "risk_off"], default="neutral")
    return out.reset_index().rename(columns={"index": "date"})


def exposure_for(regime: str) -> float:
    return {"risk_on": 1.0, "neutral": 0.70, "risk_off": 0.30}.get(regime, 0.70)


def build_trade_calendar(scores: pd.DataFrame, returns: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(set(scores["date"]).intersection(returns.index))


def pick_daily_tickers(leaders: pd.DataFrame, date: pd.Timestamp, top_themes: list[str], leaders_per_theme: int) -> list[str]:
    day = leaders[(leaders["date"].eq(date)) & (leaders["theme"].isin(top_themes))]
    day = day[day["leader_rank"] <= leaders_per_theme]
    return sorted(day["ticker"].dropna().unique())


def run_backtest(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    top_themes_n: int,
    leaders_per_theme: int,
) -> pd.DataFrame:
    dates = build_trade_calendar(scores, returns)
    regimes = build_market_regime().set_index("date")["regime"]
    score_lookup = scores.set_index(["date", "theme"])
    active: dict[str, dict] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []

    for i, date in enumerate(dates):
        regime = regimes.get(date, "neutral")
        target_exposure = exposure_for(regime)
        if active and target_exposure > 0:
            holdings = []
            for state in active.values():
                for ticker in state["tickers"]:
                    holdings.append({"ticker": ticker, "weight": target_exposure * state["weight"] / len(state["tickers"])})
            weights = pd.DataFrame(holdings).groupby("ticker")["weight"].sum()
            gross_return = float((returns.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())
            exposure = float(weights.abs().sum())
        else:
            weights = pd.Series(dtype=float)
            gross_return = 0.0
            exposure = 0.0

        updated = dict(active)
        for theme, state in list(active.items()):
            held_days = i - state["start_i"] + 1
            row = score_lookup.loc[(date, theme)] if (date, theme) in score_lookup.index else pd.Series(dtype=float)
            rank = row.get("theme_regime_rank", np.nan)
            if held_days >= HOLD_DAYS and (pd.isna(rank) or rank > 12):
                updated.pop(theme, None)

        if i % REBALANCE_EVERY_DAYS == 0:
            day_scores = scores[scores["date"].eq(date)].sort_values("theme_regime_rank")
            entries = day_scores[day_scores["theme_regime_rank"] <= top_themes_n]
            for entry in entries.itertuples(index=False):
                if entry.theme in updated:
                    continue
                tickers = pick_daily_tickers(leaders, date, [entry.theme], leaders_per_theme)
                tickers = [ticker for ticker in tickers if ticker in returns.columns]
                if tickers:
                    updated[entry.theme] = {"start_i": i + 1, "tickers": tickers, "weight": 1.0}

        if updated and target_exposure > 0:
            theme_weight = 1.0 / len(updated)
            for state in updated.values():
                state["weight"] = theme_weight
            new_holdings = []
            for state in updated.values():
                for ticker in state["tickers"]:
                    new_holdings.append({"ticker": ticker, "weight": target_exposure * state["weight"] / len(state["tickers"])})
            after_trade = pd.DataFrame(new_holdings).groupby("ticker")["weight"].sum()
        else:
            after_trade = pd.Series(dtype=float)
        turnover = float(after_trade.sub(prev_weights, fill_value=0.0).abs().sum())
        prev_weights = after_trade
        active = updated

        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append(
            {
                "date": date,
                "strategy_return": gross_return - cost,
                "gross_return": gross_return,
                "turnover": turnover,
                "exposure": exposure,
                "n_positions": int(len(weights)),
                "regime": regime,
                "active_themes": "|".join(sorted(active)),
            }
        )
    out = pd.DataFrame(rows)
    out["equity"] = (1.0 + out["strategy_return"]).cumprod()
    return out


def benchmark_returns(dates: pd.Series) -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars[bars["ticker"].isin(BENCHMARK_TICKERS)].copy()
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    return bars.pivot(index="date", columns="ticker", values="px").sort_index().reindex(dates).pct_change().fillna(0.0)


def forward_pick_stats(scores: pd.DataFrame, leaders: pd.DataFrame, returns: pd.DataFrame) -> dict[str, float]:
    score_cols = ["date", "theme", "theme_regime_rank"]
    if "is_tradable" in scores.columns:
        score_cols.append("is_tradable")
    picks = leaders.merge(scores[score_cols], on=["date", "theme"], how="inner")
    if "is_tradable" in picks.columns:
        picks = picks[picks["is_tradable"].fillna(False).astype(bool)]
    picks = picks[picks["theme_regime_rank"] <= TOP_N_THEMES].copy()
    if picks.empty:
        return {"mean_forward_5d_return": np.nan, "mean_forward_5d_return_vs_spy": np.nan}
    close_like = (1.0 + returns.fillna(0.0)).cumprod()
    future = close_like.shift(-HOLD_DAYS) / close_like - 1.0
    rows = []
    for row in picks.itertuples(index=False):
        if row.date in future.index and row.ticker in future.columns:
            rows.append({"date": row.date, "ticker": row.ticker, "forward_5d": future.at[row.date, row.ticker]})
    fwd = pd.DataFrame(rows)
    if not fwd.empty and "SPY" in returns:
        spy_curve = (1.0 + returns["SPY"].fillna(0.0)).cumprod()
        spy_fwd = spy_curve.shift(-HOLD_DAYS) / spy_curve - 1.0
        fwd = fwd.merge(spy_fwd.rename("spy_forward_5d").reset_index(), on="date", how="left")
        rel = fwd["forward_5d"] - fwd["spy_forward_5d"]
    else:
        rel = pd.Series(dtype=float)
    return {
        "mean_forward_5d_return": float(fwd["forward_5d"].mean()) if not fwd.empty else np.nan,
        "mean_forward_5d_return_vs_spy": float(rel.mean()) if len(rel) else np.nan,
    }


def _average_streak_length(selection: pd.DataFrame, flag_column: str) -> float:
    streaks: list[int] = []
    for _, frame in selection.groupby("theme"):
        frame = frame.sort_values("date")
        active = frame[flag_column].astype(bool).tolist()
        current = 0
        for is_active in active:
            if is_active:
                current += 1
            elif current:
                streaks.append(current)
                current = 0
        if current:
            streaks.append(current)
    return float(np.mean(streaks)) if streaks else 0.0


def top_theme_behavior_stats(scores: pd.DataFrame, top_n: int) -> dict[str, float]:
    tradable = scores.copy()
    if "is_tradable" in tradable.columns:
        tradable = tradable[tradable["is_tradable"].fillna(False).astype(bool)]
    tradable = tradable.dropna(subset=["theme_regime_rank"]).sort_values(["date", "theme_regime_rank"])
    top = tradable[tradable["theme_regime_rank"] <= top_n][["date", "theme"]].copy()
    top10 = tradable[tradable["theme_regime_rank"] <= 10][["date", "theme"]].copy()

    dates = sorted(top["date"].unique())
    repeats = []
    prev: set[str] | None = None
    for date in dates:
        current = set(top[top["date"].eq(date)]["theme"])
        if prev is not None and current:
            repeats.append(len(current & prev) / min(top_n, len(current)))
        prev = current

    calendar = pd.DataFrame({"date": sorted(tradable["date"].unique())})
    themes = pd.DataFrame({"theme": sorted(tradable["theme"].unique())})
    full = calendar.merge(themes, how="cross")
    full = full.merge(top.assign(in_top_theme=True), on=["date", "theme"], how="left")
    full = full.merge(top10.assign(in_top10=True), on=["date", "theme"], how="left")
    full["in_top_theme"] = full["in_top_theme"].fillna(False)
    full["in_top10"] = full["in_top10"].fillna(False)

    return {
        "avg_theme_hold_days": _average_streak_length(full, "in_top_theme"),
        "avg_rank_duration": _average_streak_length(full, "in_top10"),
        "top_theme_repeat_rate": float(np.mean(repeats)) if repeats else 0.0,
    }


def perf_stats(returns: pd.Series, equity: pd.Series, turnover: pd.Series) -> dict[str, float | None]:
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    downside_std = losers.std()
    loss_sum = abs(losers.sum())
    return {
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "sharpe": float(np.sqrt(252.0) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "sortino": float(np.sqrt(252.0) * returns.mean() / downside_std) if downside_std else 0.0,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "avg_winner": float(winners.mean()) if len(winners) else None,
        "avg_loser": float(losers.mean()) if len(losers) else None,
        "turnover": float(turnover.mean()),
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else None,
    }


def save_equity_plot(bt: pd.DataFrame, bench: pd.DataFrame) -> None:
    curve = pd.DataFrame({"theme_strategy": bt.set_index("date")["equity"]})
    for ticker in BENCHMARK_TICKERS:
        if ticker in bench:
            curve[ticker] = (1.0 + bench[ticker]).cumprod()
    curve.plot(figsize=(11, 6), title="Rule-Based Theme Rotation")
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR / "rule_based_equity_curve.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ranked themes and top leaders.")
    parser.add_argument("--top-themes", type=int, default=TOP_N_THEMES)
    parser.add_argument("--leaders-per-theme", type=int, default=LEADERS_PER_THEME)
    args = parser.parse_args()

    ensure_dirs()
    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    leaders["date"] = pd.to_datetime(leaders["date"])
    if "is_tradable" in scores.columns:
        scores = scores[scores["is_tradable"].fillna(False).astype(bool)].copy()
    returns = load_stock_returns()
    bt = run_backtest(scores, leaders, returns, args.top_themes, args.leaders_per_theme)
    bench = benchmark_returns(bt["date"])

    metrics = {"strategy": perf_stats(bt["strategy_return"], bt["equity"], bt["turnover"])}
    metrics["strategy"].update(forward_pick_stats(scores, leaders, returns))
    metrics["strategy"].update(top_theme_behavior_stats(scores, args.top_themes))
    for ticker in BENCHMARK_TICKERS:
        if ticker in bench:
            metrics[ticker] = perf_stats(bench[ticker], (1.0 + bench[ticker]).cumprod(), pd.Series(0.0, index=bench.index))

    bt.to_parquet(BACKTEST_DIR / "rule_based_theme_rotation_daily.parquet", index=False)
    (BACKTEST_DIR / "rule_based_backtest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_equity_plot(bt, bench)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
