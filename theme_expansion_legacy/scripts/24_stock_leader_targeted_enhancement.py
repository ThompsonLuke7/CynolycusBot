from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DAILY_BARS_PATH,
    OUTPUT_DIR,
    THEME_LEADERS_PATH,
    THEME_SCORES_PATH,
    TRANSACTION_COST_BPS,
    ensure_dirs,
)


OUT_DIR = OUTPUT_DIR / "deep_theme_rotation"
RESULTS_PATH = OUT_DIR / "stock_leader_targeted_results.csv"
YEARLY_PATH = OUT_DIR / "stock_leader_targeted_yearly.csv"
QUARTERLY_2025_PATH = OUT_DIR / "stock_leader_targeted_2025_quarterly.csv"
SUMMARY_PATH = OUT_DIR / "stock_leader_targeted_summary.json"

TRADING_DAYS = 252.0


def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(THEME_SCORES_PATH)
    scores["date"] = pd.to_datetime(scores["date"])
    if "is_tradable" in scores.columns:
        scores = scores[scores["is_tradable"].fillna(False).astype(bool)].copy()
    scores = scores.dropna(subset=["theme_regime_rank"]).sort_values(["date", "theme_regime_rank"]).copy()
    scores["filter_none"] = True
    scores["filter_no_breakdown"] = (scores["theme_return_5d"] > -0.03) & (scores["theme_above_20d_pct"] >= 0.35)
    scores["filter_breadth_confirm"] = (scores["theme_above_20d_pct"] >= 0.55) & (scores["theme_above_50d_pct"] >= 0.45)
    scores["filter_abs_uptrend"] = (scores["theme_return_20d"] > 0) & (scores["theme_above_20d_pct"] >= 0.50)
    return scores.replace([np.inf, -np.inf], np.nan)


def load_leaders() -> pd.DataFrame:
    leaders = pd.read_parquet(THEME_LEADERS_PATH)
    leaders["date"] = pd.to_datetime(leaders["date"])
    leaders = leaders[leaders["leader_rank"] <= 3].copy()
    return leaders.sort_values(["date", "theme", "leader_rank"])


def load_returns_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = pd.read_parquet(DAILY_BARS_PATH)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["px"] = bars["adj_close"].fillna(bars["close"])
    bars = bars.sort_values(["ticker", "date"])
    bars["return_1d"] = bars.groupby("ticker")["px"].pct_change()
    returns = bars.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    prices = bars.pivot(index="date", columns="ticker", values="px").sort_index()
    return returns, prices


def build_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    for ticker in ["SPY", "QQQ"]:
        out[f"{ticker.lower()}_above_200dma"] = prices[ticker] > prices[ticker].rolling(200, min_periods=100).mean()
        out[f"{ticker.lower()}_return_5d"] = prices[ticker].pct_change(5)
        out[f"{ticker.lower()}_return_20d"] = prices[ticker].pct_change(20)
        out[f"{ticker.lower()}_drawdown_63d"] = prices[ticker] / prices[ticker].rolling(63, min_periods=20).max() - 1.0

    excluded = {"SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "USO", "SMH", "XLK", "XLF", "XLV", "XLE"}
    stock_cols = [col for col in prices.columns if col not in excluded]
    sma200 = prices[stock_cols].rolling(200, min_periods=100).mean()
    out["market_breadth"] = (prices[stock_cols] > sma200).mean(axis=1)
    out["risk_off"] = (
        ((~out["spy_above_200dma"].fillna(False)) & (~out["qqq_above_200dma"].fillna(False)))
        | (out["market_breadth"] < 0.40)
        | ((out["spy_drawdown_63d"] < -0.12) & (out["spy_return_20d"] < -0.05))
    )
    out["risk_on"] = out["spy_above_200dma"].fillna(False) & out["qqq_above_200dma"].fillna(False) & (out["market_breadth"] > 0.55)
    out["regime"] = np.select([out["risk_on"], out["risk_off"]], ["risk_on", "risk_off"], default="neutral")
    out["qqq_recovery"] = (
        (out["spy_return_5d"] > 0.035)
        & (out["qqq_return_5d"] > 0.040)
        & (out["spy_drawdown_63d"].shift(5) < -0.08)
    )
    return out


def exposure_for(mode: str, regime: str, risk_off: bool) -> float:
    if mode == "full":
        return 1.0
    if mode == "scaled":
        return {"risk_on": 1.0, "neutral": 0.70, "risk_off": 0.30}.get(regime, 0.70)
    if mode == "cash_risk_off":
        return 0.0 if risk_off else 1.0
    raise ValueError(f"unknown exposure mode {mode}")


def perf_stats(returns: pd.Series, turnover: pd.Series) -> dict[str, float]:
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = max(len(returns) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    loss_sum = abs(losers.sum())
    final_equity = float(equity.iloc[-1]) if len(equity) else 1.0
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 else np.nan
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    return {
        "total_return": final_equity - 1.0,
        "cagr": float(cagr) if pd.notna(cagr) else np.nan,
        "sharpe": float(np.sqrt(TRADING_DAYS) * returns.mean() / returns.std()) if returns.std() else 0.0,
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if pd.notna(cagr) and max_drawdown < 0 else np.nan,
        "hit_rate": float((returns > 0).mean()),
        "profit_factor": float(winners.sum() / loss_sum) if loss_sum > 0 else np.nan,
        "turnover": float(turnover.fillna(0.0).mean()),
    }


def pick_tickers(leaders_by_date_theme: dict[tuple[pd.Timestamp, str], list[str]], date: pd.Timestamp, theme: str) -> list[str]:
    return leaders_by_date_theme.get((date, theme), [])


def run_strategy(
    *,
    label: str,
    scores: pd.DataFrame,
    leaders_by_date_theme: dict[tuple[pd.Timestamp, str], list[str]],
    returns: pd.DataFrame,
    market: pd.DataFrame,
    top_n: int,
    exit_rank: int,
    min_hold: int,
    filter_name: str,
    exposure_mode: str,
    exit_on_filter_fail: bool,
    qqq_recovery_fallback: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    dates = sorted(set(scores["date"]).intersection(returns.index))
    score_lookup = scores.set_index(["date", "theme"])
    scores_by_date = {date: day.sort_values("theme_regime_rank") for date, day in scores.groupby("date", sort=False)}
    active: dict[str, dict[str, object]] = {}
    prev_weights = pd.Series(dtype=float)
    rows = []
    filter_col = f"filter_{filter_name}"

    for i, date in enumerate(dates):
        gross_return = float((returns.loc[date].reindex(prev_weights.index).fillna(0.0) * prev_weights).sum()) if not prev_weights.empty else 0.0
        feature = market.loc[date] if date in market.index else pd.Series(dtype=object)

        updated = dict(active)
        for theme, state in list(active.items()):
            row = score_lookup.loc[(date, theme)] if (date, theme) in score_lookup.index else pd.Series(dtype=object)
            rank = row.get("theme_regime_rank", np.nan)
            passes_filter = bool(row.get(filter_col, False)) if filter_col in row.index else False
            held_days = i - int(state["start_i"]) + 1
            exit_filter = exit_on_filter_fail and not passes_filter
            if held_days >= min_hold and (pd.isna(rank) or rank > exit_rank or exit_filter):
                updated.pop(theme, None)

        day = scores_by_date[date]
        entries = day[day[filter_col].fillna(False)].head(top_n)
        for entry in entries.itertuples(index=False):
            theme = str(entry.theme)
            if theme in updated:
                continue
            tickers = [ticker for ticker in pick_tickers(leaders_by_date_theme, date, theme) if ticker in returns.columns]
            if tickers:
                updated[theme] = {"start_i": i + 1, "tickers": tickers}

        risk_off = bool(feature.get("risk_off", False))
        regime = str(feature.get("regime", "neutral"))
        exposure = exposure_for(exposure_mode, regime, risk_off)
        if qqq_recovery_fallback and bool(feature.get("qqq_recovery", False)) and "QQQ" in returns.columns:
            next_weights = pd.Series({"QQQ": 1.0}, dtype=float)
        elif updated and exposure > 0:
            holdings = []
            theme_weight = exposure / len(updated)
            for theme, state in updated.items():
                tickers = state["tickers"]
                for ticker in tickers:
                    holdings.append({"ticker": ticker, "weight": theme_weight / len(tickers)})
            next_weights = pd.DataFrame(holdings).groupby("ticker")["weight"].sum()
        else:
            next_weights = pd.Series(dtype=float)

        turnover = float(next_weights.sub(prev_weights, fill_value=0.0).abs().sum())
        cost = turnover * TRANSACTION_COST_BPS / 10_000.0
        rows.append(
            {
                "date": date,
                "strategy_return": gross_return - cost,
                "turnover": turnover,
                "n_active_themes": len(updated),
                "n_positions": int(len(next_weights)),
                "active_themes": "|".join(sorted(updated)),
                "fallback_active": bool(qqq_recovery_fallback and "QQQ" in next_weights.index),
            }
        )
        active = updated
        prev_weights = next_weights

    daily = pd.DataFrame(rows)
    daily["equity"] = (1.0 + daily["strategy_return"].fillna(0.0)).cumprod()
    stats = perf_stats(daily["strategy_return"], daily["turnover"])
    return {
        "label": label,
        "top_n": top_n,
        "exit_rank": exit_rank,
        "min_hold": min_hold,
        "filter": filter_name,
        "exposure_mode": exposure_mode,
        "exit_on_filter_fail": exit_on_filter_fail,
        "qqq_recovery_fallback": qqq_recovery_fallback,
        **stats,
    }, daily


def period_stats(daily: pd.DataFrame, label: str, freq: str) -> pd.DataFrame:
    frame = daily.copy()
    frame["period"] = frame["date"].dt.to_period(freq).astype(str)
    rows = []
    for period, group in frame.groupby("period", sort=True):
        rows.append({"period": period, "strategy": label, **perf_stats(group["strategy_return"], group["turnover"]), "days": int(len(group))})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    leaders = load_leaders()
    returns, prices = load_returns_prices()
    market = build_market_features(prices)
    leaders_by_date_theme = {
        (date, theme): frame.sort_values("leader_rank")["ticker"].dropna().tolist()
        for (date, theme), frame in leaders.groupby(["date", "theme"], sort=False)
    }

    configs = [
        ("existing_like_top3_scaled", 3, 12, 5, "none", "scaled", False, False),
        ("top5_leaders_full", 5, 12, 5, "none", "full", False, False),
        ("top5_leaders_cash_risk_off", 5, 12, 5, "none", "cash_risk_off", False, False),
        ("top5_leaders_no_breakdown_full", 5, 12, 5, "no_breakdown", "full", True, False),
        ("top5_leaders_no_breakdown_cash_risk_off", 5, 12, 5, "no_breakdown", "cash_risk_off", True, False),
        ("top5_leaders_breadth_cash_risk_off", 5, 12, 5, "breadth_confirm", "cash_risk_off", True, False),
        ("top5_leaders_no_breakdown_qqq_recovery", 5, 12, 5, "no_breakdown", "full", True, True),
        ("top5_leaders_abs_uptrend_cash_risk_off", 5, 12, 5, "abs_uptrend", "cash_risk_off", True, False),
    ]

    rows = []
    yearly = []
    quarterly = []
    for label, top_n, exit_rank, min_hold, filter_name, exposure_mode, exit_on_filter_fail, qqq_fallback in configs:
        row, daily = run_strategy(
            label=label,
            scores=scores,
            leaders_by_date_theme=leaders_by_date_theme,
            returns=returns,
            market=market,
            top_n=top_n,
            exit_rank=exit_rank,
            min_hold=min_hold,
            filter_name=filter_name,
            exposure_mode=exposure_mode,
            exit_on_filter_fail=exit_on_filter_fail,
            qqq_recovery_fallback=qqq_fallback,
        )
        rows.append(row)
        daily.to_parquet(OUT_DIR / f"{label}_daily.parquet", index=False)
        yearly.append(period_stats(daily, label, "Y"))
        quarterly.append(period_stats(daily, label, "Q"))

    results = pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False)
    yearly_out = pd.concat(yearly, ignore_index=True)
    quarterly_out = pd.concat(quarterly, ignore_index=True)
    quarterly_out = quarterly_out[quarterly_out["period"].str.startswith("2025")].copy()
    results.to_csv(RESULTS_PATH, index=False)
    yearly_out.to_csv(YEARLY_PATH, index=False)
    quarterly_out.to_csv(QUARTERLY_2025_PATH, index=False)

    summary = {
        "top_results": results.to_dict(orient="records"),
        "quarterly_2025": quarterly_out.to_dict(orient="records"),
        "output_files": {
            "results": str(RESULTS_PATH),
            "yearly": str(YEARLY_PATH),
            "quarterly_2025": str(QUARTERLY_2025_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("targeted stock leader configs")
    print(results[["label", "cagr", "sharpe", "max_drawdown", "turnover"]].to_string(index=False))
    print("\n2025Q2")
    print(
        quarterly_out[quarterly_out["period"].eq("2025Q2")][
            ["strategy", "total_return", "sharpe", "max_drawdown"]
        ].to_string(index=False)
    )
    print(f"\nsaved stock leader targeted tests -> {OUT_DIR}")


if __name__ == "__main__":
    main()
