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

OUT_DIR = OUTPUT_DIR / "market_regime_tests"


def load_bars() -> pd.DataFrame:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    return bars.sort_values(["ticker", "date"])


def download_vix(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download("^VIX", start=start.strftime("%Y-%m-%d"), end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=False, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame(columns=["date", "vix"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw.reset_index()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    px_col = "adj_close" if "adj_close" in out.columns else "close"
    return out[["date", px_col]].rename(columns={px_col: "vix"})


def returns_wide(bars: pd.DataFrame) -> pd.DataFrame:
    wide = bars.pivot(index="date", columns="ticker", values="px").sort_index()
    return wide.pct_change().fillna(0.0)


def build_market_regime(bars: pd.DataFrame) -> pd.DataFrame:
    wide = bars.pivot(index="date", columns="ticker", values="px").sort_index()
    out = pd.DataFrame(index=wide.index)
    for ticker in ["SPY", "QQQ"]:
        out[f"{ticker.lower()}_above_200dma"] = wide[ticker] > wide[ticker].rolling(200, min_periods=100).mean()
    out["tlt_return_20d"] = wide["TLT"].pct_change(20) if "TLT" in wide else np.nan

    stock_cols = [c for c in wide.columns if c not in {"SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "SMH", "XLK", "XLF", "XLV", "XLE"}]
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
    return out.reset_index()


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


def exposure_for(regime: str, mode: str) -> float:
    if mode == "baseline":
        return 1.0
    if mode == "risk_on_only":
        return 1.0 if regime == "risk_on" else 0.0
    if mode == "scaled":
        return {"risk_on": 1.0, "neutral": 0.70, "risk_off": 0.30}.get(regime, 0.70)
    return 1.0


def simulate(
    scores: pd.DataFrame,
    leaders: pd.DataFrame,
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    mode: str,
    enter_min_rank: int,
    enter_max_rank: int,
    exit_rank: int = 12,
    min_hold_days: int = 5,
) -> pd.DataFrame:
    dates = sorted(set(scores["date"]).intersection(returns.index))
    score_lookup = scores.set_index(["date", "theme"])
    regime_lookup = regimes.set_index("date")["regime"]
    active: dict[str, dict] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []

    for i, date in enumerate(dates):
        regime = regime_lookup.get(date, "neutral")
        target_exposure = exposure_for(regime, mode)

        gross = 0.0
        if active and target_exposure > 0:
            holdings = []
            for state in active.values():
                for ticker in state["tickers"]:
                    holdings.append({"ticker": ticker, "weight": target_exposure * state["weight"] / len(state["tickers"])})
            weights = pd.DataFrame(holdings).groupby("ticker")["weight"].sum()
            gross = float((returns.loc[date].reindex(weights.index).fillna(0.0) * weights).sum())

        updated = dict(active)
        for theme, state in list(active.items()):
            held_days = i - state["start_i"] + 1
            row = score_lookup.loc[(date, theme)] if (date, theme) in score_lookup.index else pd.Series(dtype=float)
            rank = row.get("theme_regime_rank", np.nan)
            if held_days >= min_hold_days and (pd.isna(rank) or rank > exit_rank):
                updated.pop(theme, None)

        day = scores[scores["date"].eq(date)].sort_values("theme_regime_rank")
        entries = day[(day["theme_regime_rank"] >= enter_min_rank) & (day["theme_regime_rank"] <= enter_max_rank)]
        for entry in entries.itertuples(index=False):
            if entry.theme in updated:
                continue
            tickers = [t for t in pick_theme_leaders(leaders, date, entry.theme) if t in returns.columns]
            if not tickers:
                continue
            updated[entry.theme] = {"start_i": i + 1, "tickers": tickers, "weight": 1.0}

        if updated and target_exposure > 0:
            theme_weight = 1.0 / len(updated)
            for state in updated.values():
                state["weight"] = theme_weight
            new_holdings = []
            for state in updated.values():
                for ticker in state["tickers"]:
                    new_holdings.append({"ticker": ticker, "weight": target_exposure * state["weight"] / len(state["tickers"])})
            new_weights = pd.DataFrame(new_holdings).groupby("ticker")["weight"].sum()
        else:
            new_weights = pd.Series(dtype=float)

        turnover = float(new_weights.sub(prev_weights, fill_value=0.0).abs().sum())
        prev_weights = new_weights
        active = updated
        rows.append(
            {
                "date": date,
                "mode": mode,
                "regime": regime,
                "strategy_return": gross - turnover * TRANSACTION_COST_BPS / 10_000.0,
                "turnover": turnover,
                "exposure": target_exposure,
                "active_themes": "|".join(sorted(active)),
            }
        )
    return pd.DataFrame(rows)


def period_2022_stats(bt: pd.DataFrame) -> dict[str, float | None]:
    sub = bt[(bt["date"] >= pd.Timestamp("2022-01-01")) & (bt["date"] <= pd.Timestamp("2022-12-31"))]
    if sub.empty:
        return {"cagr_2022": None, "sharpe_2022": None, "max_dd_2022": None}
    stats = perf_stats(sub["strategy_return"], sub["turnover"])
    return {"cagr_2022": stats["cagr"], "sharpe_2022": stats["sharpe"], "max_dd_2022": stats["max_dd"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test market regime exposure scaling and delayed theme momentum.")
    args = parser.parse_args()
    _ = args
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bars = load_bars()
    returns = returns_wide(bars)
    regimes = build_market_regime(bars)
    regimes.to_csv(OUT_DIR / "market_regime_daily.csv", index=False)

    scores = pd.read_parquet(THEME_SCORES_PATH)
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    for frame in (scores, leaders):
        frame["date"] = pd.to_datetime(frame["date"])
    scores = tradable_scores(scores)

    cases = [
        ("v3_baseline", "baseline", 1, 3),
        ("v3_risk_on_only", "risk_on_only", 1, 3),
        ("v3_scaled_exposure", "scaled", 1, 3),
        ("theme_delayed_rank_4_8", "baseline", 4, 8),
    ]
    rows = []
    curves = {}
    for case_name, mode, enter_min, enter_max in cases:
        bt = simulate(scores, leaders, returns, regimes, mode=mode, enter_min_rank=enter_min, enter_max_rank=enter_max)
        bt.to_csv(OUT_DIR / f"{case_name}_daily.csv", index=False)
        stats = perf_stats(bt["strategy_return"], bt["turnover"])
        stats.update(period_2022_stats(bt))
        rows.append({"case": case_name, **stats})
        curves[case_name] = (1.0 + bt.set_index("date")["strategy_return"]).cumprod()

    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out.to_csv(OUT_DIR / "market_regime_test_results.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_DIR / "market_regime_equity_curves.csv")
    (OUT_DIR / "market_regime_test_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"saved market regime tests -> {OUT_DIR}")


if __name__ == "__main__":
    main()
