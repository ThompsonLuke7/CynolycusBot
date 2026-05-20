"""Minute-by-minute historical replay engine."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from momentum_scalper.execution.entry_policy import evaluate_entry
from momentum_scalper.execution.exit_policy import simulate_exit
from momentum_scalper.features.build_features import build_features_for_snapshot
from momentum_scalper.rankers.rule_ranker import top_ranked_setups
from momentum_scalper.scanners.historical_scanner import load_bars_for_day, reconstruct_premarket_scanner
from momentum_scalper.utils.io import add_session_columns, write_parquet


def replay_day(day: str, top_n: int = 5) -> pd.DataFrame:
    bars = add_session_columns(load_bars_for_day(day))
    if bars.empty:
        return pd.DataFrame()
    trades: list[dict] = []
    scanner = reconstruct_premarket_scanner(bars)
    for ts, snapshot in scanner.groupby("timestamp", sort=True):
        features = build_features_for_snapshot(snapshot, bars[bars["timestamp"] <= ts])
        ranked = top_ranked_setups(features, top_n=top_n)
        for _, row in ranked.iterrows():
            signal = evaluate_entry(row)
            if not signal.should_enter:
                continue
            hist = bars[(bars["ticker"].eq(row["ticker"])) & (bars["timestamp"] <= ts)].sort_values("timestamp")
            if hist.empty:
                continue
            entry_price = float(hist.iloc[-1]["close"])
            forward = bars[(bars["ticker"].eq(row["ticker"])) & (bars["timestamp"] > ts) & (bars["timestamp"] <= ts + pd.Timedelta(minutes=30))]
            exit_result = simulate_exit(forward, entry_price, ts)
            if exit_result is None:
                continue
            trades.append(
                {
                    "entry_timestamp": ts,
                    "ticker": row["ticker"],
                    "pattern": signal.pattern,
                    "rank": row["rank"],
                    "score": row["score"],
                    "entry_price": entry_price,
                    "exit_timestamp": exit_result.exit_timestamp,
                    "exit_price": exit_result.exit_price,
                    "exit_reason": exit_result.reason,
                    "pnl_pct": exit_result.pnl_pct,
                    "MFE": exit_result.mfe_pct,
                    "MAE": exit_result.mae_pct,
                    "right_tail_capture": exit_result.right_tail_capture,
                    "average_giveback": exit_result.average_giveback,
                }
            )
    return pd.DataFrame(trades)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one historical day")
    parser.add_argument("--day", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trades = replay_day(args.day)
    if args.output:
        write_parquet(trades, args.output)
    print(trades.tail(50).to_string(index=False) if not trades.empty else "no trades")


if __name__ == "__main__":
    main()
