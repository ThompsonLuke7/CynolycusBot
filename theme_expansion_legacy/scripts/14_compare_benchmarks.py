from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

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

OUT_DIR = OUTPUT_DIR / "benchmark_comparison"
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


def load_bars() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    return bars.sort_values(["ticker", "date"])


def download_etf(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=False, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw.reset_index()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if "adj_close" not in out and "close" in out:
        out["adj_close"] = out["close"]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = ticker
    out["px"] = out["adj_close"].fillna(out["close"])
    return out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "px"]]


def ensure_sector_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = [t for t in SECTOR_ETFS if t not in set(bars["ticker"])]
    if not missing:
        return bars
    start = bars["date"].min()
    end = bars["date"].max()
    frames = [bars]
    for ticker in missing:
        df = download_etf(ticker, start, end)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"]).sort_values(["ticker", "date"])


def returns_wide(bars: pd.DataFrame) -> pd.DataFrame:
    wide = bars.pivot(index="date", columns="ticker", values="px").sort_index()
    return wide.pct_change().fillna(0.0)


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


def buy_hold(ret: pd.DataFrame, ticker: str) -> tuple[pd.Series, pd.Series]:
    return ret[ticker].fillna(0.0), pd.Series(0.0, index=ret.index)


def run_rank_hysteresis(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    id_col: str,
    rank_col: str,
    enter_rank: int,
    exit_rank: int,
    min_hold_days: int,
    max_names: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    dates = sorted(set(signal["date"]).intersection(ret.index))
    lookup = signal.set_index(["date", id_col])
    active: dict[str, dict] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []
    turnover_rows = []

    for i, date in enumerate(dates):
        if active:
            weights = pd.Series({k: v["weight"] for k, v in active.items()})
            weights = weights / weights.abs().sum()
            gross = float((ret.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())
        else:
            gross = 0.0

        updated = dict(active)
        for name, state in list(active.items()):
            held_days = i - state["start_i"] + 1
            row = lookup.loc[(date, name)] if (date, name) in lookup.index else pd.Series(dtype=float)
            rank = row.get(rank_col, np.nan)
            if held_days >= min_hold_days and (pd.isna(rank) or rank > exit_rank):
                updated.pop(name, None)

        entries = signal[(signal["date"].eq(date)) & (signal[rank_col] <= enter_rank)].sort_values(rank_col)
        if max_names is not None:
            entries = entries.head(max_names)
        for entry in entries.itertuples(index=False):
            name = getattr(entry, id_col)
            if name in updated or name not in ret.columns:
                continue
            updated[name] = {"start_i": i + 1, "weight": 1.0}

        if updated:
            w = 1.0 / len(updated)
            new_weights = pd.Series({k: w for k in updated})
        else:
            new_weights = pd.Series(dtype=float)
        turnover = float(new_weights.sub(prev_weights, fill_value=0.0).abs().sum())
        prev_weights = new_weights
        active = updated
        rows.append(gross - turnover * TRANSACTION_COST_BPS / 10_000.0)
        turnover_rows.append(turnover)

    return pd.Series(rows, index=dates), pd.Series(turnover_rows, index=dates)


def current_best_theme(ret: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    leaders["date"] = pd.to_datetime(leaders["date"])
    scores = scores[scores["is_tradable"].fillna(False).astype(bool)].dropna(subset=["theme_regime_rank"])
    dates = sorted(set(scores["date"]).intersection(ret.index))
    lookup = scores.set_index(["date", "theme"])
    active: dict[str, dict] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []
    turnover_rows = []

    for i, date in enumerate(dates):
        holdings = []
        for theme, state in active.items():
            for ticker in state["tickers"]:
                holdings.append({"ticker": ticker, "weight": state["weight"] / len(state["tickers"])})
        if holdings:
            weights = pd.DataFrame(holdings).groupby("ticker")["weight"].sum()
            weights = weights / weights.abs().sum()
            gross = float((ret.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())
        else:
            gross = 0.0

        updated = dict(active)
        for theme, state in list(active.items()):
            held_days = i - state["start_i"] + 1
            row = lookup.loc[(date, theme)] if (date, theme) in lookup.index else pd.Series(dtype=float)
            rank = row.get("theme_regime_rank", np.nan)
            if held_days >= 5 and (pd.isna(rank) or rank > 12):
                updated.pop(theme, None)

        entries = scores[(scores["date"].eq(date)) & (scores["theme_regime_rank"] <= 3)].sort_values("theme_regime_rank")
        for entry in entries.itertuples(index=False):
            if entry.theme in updated:
                continue
            day = leaders[(leaders["date"].eq(date)) & (leaders["theme"].eq(entry.theme))]
            tickers = day.sort_values("leader_rank").head(LEADERS_PER_THEME)["ticker"].dropna().tolist()
            tickers = [t for t in tickers if t in ret.columns]
            if not tickers:
                continue
            updated[entry.theme] = {"start_i": i + 1, "tickers": tickers, "weight": 1.0}

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
        rows.append(gross - turnover * TRANSACTION_COST_BPS / 10_000.0)
        turnover_rows.append(turnover)

    return pd.Series(rows, index=dates), pd.Series(turnover_rows, index=dates)


def stock_momentum_signal(bars: pd.DataFrame) -> pd.DataFrame:
    stocks = bars[~bars["ticker"].isin(SECTOR_ETFS + ["SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "SMH", "XLK", "XLF", "XLV", "XLE"])].copy()
    wide = stocks.pivot(index="date", columns="ticker", values="px").sort_index()
    mom = wide.pct_change(20)
    signal = mom.stack().rename("momentum_20d").reset_index()
    signal["momentum_rank"] = signal.groupby("date")["momentum_20d"].rank(ascending=False, method="first")
    return signal.rename(columns={"ticker": "name"})


def sector_signal(bars: pd.DataFrame) -> pd.DataFrame:
    wide = bars[bars["ticker"].isin(SECTOR_ETFS)].pivot(index="date", columns="ticker", values="px").sort_index()
    mom = wide.pct_change(20)
    signal = mom.stack().rename("momentum_20d").reset_index()
    signal["momentum_rank"] = signal.groupby("date")["momentum_20d"].rank(ascending=False, method="first")
    return signal.rename(columns={"ticker": "name"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare best theme regime strategy against simple benchmarks.")
    args = parser.parse_args()
    _ = args
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = ensure_sector_bars(load_bars())
    ret = returns_wide(bars)

    rows = []
    curves = {}
    for name, (series, turnover) in {
        "theme_regime_best": current_best_theme(ret),
        "SPY": buy_hold(ret, "SPY"),
        "QQQ": buy_hold(ret, "QQQ"),
        "top9_stock_momentum": run_rank_hysteresis(stock_momentum_signal(bars), ret, id_col="name", rank_col="momentum_rank", enter_rank=9, exit_rank=18, min_hold_days=5, max_names=9),
        "top3_sector_etf_rotation": run_rank_hysteresis(sector_signal(bars), ret, id_col="name", rank_col="momentum_rank", enter_rank=3, exit_rank=6, min_hold_days=5, max_names=3),
    }.items():
        stats = perf_stats(series, turnover)
        rows.append({"strategy": name, **stats})
        curves[name] = (1.0 + series).cumprod()

    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out.to_csv(OUT_DIR / "benchmark_comparison.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_DIR / "benchmark_equity_curves.csv")
    (OUT_DIR / "benchmark_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"saved benchmark comparison -> {OUT_DIR}")


if __name__ == "__main__":
    main()
