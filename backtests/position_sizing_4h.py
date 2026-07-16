"""Fixed-protocol 4H position-sizing comparison.

Uses the already frozen, validation-selected test trades for the Momentum and
HTF base-model modules.  It deliberately changes *only* sizing: trade
selection, entry/exit prices, and the validation-selected policies are held
fixed.  Meta Ranker is not included because it has no analogous frozen-test
trade ledger yet; do not infer a Meta sizing recommendation from this run.

Schemes:
  - fixed_100_shares: current live price-level-dependent sizing;
  - equal_notional: fixed dollar exposure per trade;
  - vol_scaled: fixed stop-risk budget, capped at equal-notional exposure.

The equity curve realizes P&L at each recorded exit.  Entries are accepted in
a deterministic timestamp/strategy/ticker order only while their aggregate
gross entry notional remains within the configured capital cap; concurrent
same-ticker trades are rejected.  It remains a conservative sizing comparison,
not a live-policy change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "position_sizing_4h"
SOURCES = {
    "momentum": ROOT / "strategies/momentum_expansion/backtest/results/family_compare_clean",
    "htf": ROOT / "strategies/multi_ticker_swing_htf/backtest/results/family_compare_clean",
}


def _frozen_trades(strategy: str) -> tuple[pd.DataFrame, dict]:
    source = SOURCES[strategy]
    summary = json.loads((source / "comparison_summary_clean.json").read_text())
    row = summary["deployed_winner_frozen_test"]
    family, seed = str(row["family"]), int(row["seed"])
    path = source / f"{family}_s{seed}_frozen_test_trades.parquet"
    trades = pd.read_parquet(path).copy()
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True)
    trades["strategy"] = strategy
    trades["sl_atr_mult"] = float(row["sl_atr_mult"])
    return trades, {"path": str(path), "family": family, "seed": seed, "policy": row}


def _shares_for(
    trades: pd.DataFrame,
    *,
    scheme: str,
    fixed_shares: int,
    equal_notional: float,
    risk_budget: float,
) -> np.ndarray:
    entry = trades["entry_price"].to_numpy(float)
    if scheme == "fixed_100_shares":
        return np.full(len(trades), fixed_shares, dtype=int)
    if scheme == "equal_notional":
        return np.floor(equal_notional / entry).clip(0).astype(int)
    if scheme == "vol_scaled":
        stop_distance = trades["sl_atr_mult"].to_numpy(float) * trades["atr_at_entry"].to_numpy(float)
        risk_shares = np.divide(risk_budget, stop_distance, out=np.zeros(len(trades)), where=stop_distance > 0)
        notional_shares = equal_notional / entry
        return np.floor(np.minimum(risk_shares, notional_shares)).clip(0).astype(int)
    raise ValueError(f"unknown scheme {scheme}")


def _max_concurrent_notional(trades: pd.DataFrame) -> float:
    events = []
    for r in trades.itertuples():
        events.append((r.entry_ts, 1, float(r.entry_notional)))
        events.append((r.exit_ts, 0, -float(r.entry_notional)))  # exit first at a shared timestamp
    gross = peak = 0.0
    for _ts, _kind, change in sorted(events, key=lambda x: (x[0], x[1])):
        gross += change
        peak = max(peak, gross)
    return peak


def apply_portfolio_cap(trades: pd.DataFrame, *, max_gross_notional: float) -> pd.DataFrame:
    """Accept a deterministic, no-duplicate-ticker subset within a gross cap.

    The frozen trade ledgers intentionally retain every scored signal for
    signal-quality evaluation.  A live portfolio cannot hold all of them at
    once, so use the same ordering for every sizing scheme and release gross
    exposure at each recorded exit before evaluating new entries.
    """
    ordered = trades.sort_values(["entry_ts", "strategy", "ticker", "score"], ascending=[True, True, True, False]).copy()
    active: list[dict] = []
    accepted: list[bool] = []
    reasons: list[str] = []
    gross = 0.0
    for row in ordered.itertuples():
        still_open = []
        for position in active:
            if position["exit_ts"] <= row.entry_ts:
                gross -= position["notional"]
            else:
                still_open.append(position)
        active = still_open
        if any(position["ticker"] == row.ticker for position in active):
            accepted.append(False)
            reasons.append("ticker_already_open")
            continue
        if gross + row.entry_notional > max_gross_notional + 1e-9:
            accepted.append(False)
            reasons.append("gross_cap")
            continue
        active.append({"ticker": row.ticker, "exit_ts": row.exit_ts, "notional": row.entry_notional})
        gross += row.entry_notional
        accepted.append(True)
        reasons.append("accepted")
    ordered["accepted"] = accepted
    ordered["portfolio_reason"] = reasons
    return ordered


def _metrics(trades: pd.DataFrame, *, initial_capital: float) -> dict:
    t = trades.sort_values("exit_ts").reset_index(drop=True)
    equity = initial_capital + t["net_pnl"].cumsum()
    drawdown = (equity - equity.cummax()) / equity.cummax()
    ticker = t.groupby("ticker")["net_pnl"].sum()
    gross_abs = float(ticker.abs().sum())
    return {
        "trades": int(len(t)),
        "win_rate": float((t["net_pnl"] > 0).mean()),
        "net_pnl": float(t["net_pnl"].sum()),
        "total_return_pct": float(t["net_pnl"].sum() / initial_capital * 100),
        "max_realized_dd_pct": float(drawdown.min() * 100),
        "profit_factor": float(t.loc[t.net_pnl > 0, "net_pnl"].sum() /
                               max(-t.loc[t.net_pnl <= 0, "net_pnl"].sum(), 1e-12)),
        "mean_entry_notional": float(t["entry_notional"].mean()),
        "p95_entry_notional": float(t["entry_notional"].quantile(0.95)),
        "max_concurrent_notional": _max_concurrent_notional(t),
        "max_concurrent_gross_pct": float(_max_concurrent_notional(t) / initial_capital * 100),
        "top_ticker_abs_pnl_share": float(ticker.abs().max() / gross_abs) if gross_abs else 0.0,
    }


def size_trades(
    trades: pd.DataFrame,
    *,
    scheme: str,
    fixed_shares: int,
    equal_notional: float,
    risk_budget: float,
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    out = trades.copy()
    out["shares"] = _shares_for(
        out, scheme=scheme, fixed_shares=fixed_shares,
        equal_notional=equal_notional, risk_budget=risk_budget,
    )
    out = out[out["shares"] > 0].copy()
    out["entry_notional"] = out["shares"] * out["entry_price"]
    out["gross_pnl"] = out["shares"] * out["entry_price"] * out["pnl_pct"]
    out["round_trip_cost"] = out["entry_notional"] * round_trip_cost_bps / 10_000.0
    out["net_pnl"] = out["gross_pnl"] - out["round_trip_cost"]
    return out


def run(
    *,
    initial_capital: float = 100_000.0,
    fixed_shares: int = 100,
    equal_notional: float = 10_000.0,
    vol_risk_pct: float = 0.25,
    round_trip_cost_bps: float = 20.0,
    max_gross_leverage: float = 1.0,
) -> pd.DataFrame:
    source_meta: dict[str, dict] = {}
    all_trades = []
    for strategy in SOURCES:
        trades, meta = _frozen_trades(strategy)
        source_meta[strategy] = meta
        all_trades.append(trades)
    base = pd.concat(all_trades, ignore_index=True)
    risk_budget = initial_capital * vol_risk_pct / 100.0
    rows = []
    for scheme in ("fixed_100_shares", "equal_notional", "vol_scaled"):
        sized = size_trades(
            base, scheme=scheme, fixed_shares=fixed_shares,
            equal_notional=equal_notional, risk_budget=risk_budget,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        capped = apply_portfolio_cap(sized, max_gross_notional=initial_capital * max_gross_leverage)
        capped.to_parquet(OUT / f"{scheme}_trades.parquet", index=False)
        for scope, subset in [("momentum", capped[capped.strategy == "momentum"]),
                              ("htf", capped[capped.strategy == "htf"]),
                              ("combined", capped)]:
            accepted = subset[subset.accepted].copy()
            reasons = subset["portfolio_reason"].value_counts()
            rows.append({
                "scope": scope, "scheme": scheme, "candidate_trades": int(len(subset)),
                "accepted_trades": int(len(accepted)),
                "gross_cap_skips": int(reasons.get("gross_cap", 0)),
                "ticker_overlap_skips": int(reasons.get("ticker_already_open", 0)),
                **_metrics(accepted, initial_capital=initial_capital),
            })
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "summary.csv", index=False)
    (OUT / "method.json").write_text(json.dumps({
        "method": "same frozen test trades under three sizing rules; no policy selection performed",
        "initial_capital": initial_capital,
        "fixed_shares": fixed_shares,
        "equal_notional": equal_notional,
        "vol_risk_pct": vol_risk_pct,
        "risk_budget_dollars": risk_budget,
        "round_trip_cost_bps": round_trip_cost_bps,
        "max_gross_leverage": max_gross_leverage,
        "sources": source_meta,
        "limitations": [
            "realized-equity curve; open positions are not marked to market between entry and exit",
            "gross-cap and no-duplicate-ticker constraints are a conservative proxy, not a broker-margin model",
            "Meta Ranker excluded until it has an equivalent frozen-test trade ledger",
        ],
    }, indent=2, default=str))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen 4H trade outcomes under sizing rules.")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--fixed-shares", type=int, default=100)
    parser.add_argument("--equal-notional", type=float, default=10_000.0)
    parser.add_argument("--vol-risk-pct", type=float, default=0.25)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--max-gross-leverage", type=float, default=1.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = run(**vars(args))
    print(result.round({"win_rate": 4, "net_pnl": 2, "total_return_pct": 3,
                        "max_realized_dd_pct": 3, "profit_factor": 3,
                        "mean_entry_notional": 2, "p95_entry_notional": 2,
                        "max_concurrent_notional": 2, "max_concurrent_gross_pct": 2,
                        "top_ticker_abs_pnl_share": 4}).to_string(index=False))


if __name__ == "__main__":
    main()
