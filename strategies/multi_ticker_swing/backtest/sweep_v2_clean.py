"""
Val-select / test-freeze patch for sweep_v2 (see research/capstone/leakage_audit.md §1.4).

sweep_v2.run_sweep() picks the best of 180 (entry_threshold x confirm_bars x
exit_strategy) combos BY SHARPE on the SAME split it then reports grouped
metrics for (default --split test) — the advisor doc's 62.6%/60.0% long/short
win-rate figures come from that self-selected result. This script fixes the
selection: sweep the full grid on the VAL split, pick the Sharpe-best combo
there, then freeze that exact combo and simulate it, untouched, on the TEST
split.

Also depends on the sweep_v2.py raw-loader fix (this session): the module's
local load_raw_30m/load_raw_5m assumed "timestamp" was always a column, but
most raw 30m/5m caches now store it as a DatetimeIndex — silently dropping
most tickers from every prior sweep_v2 run (n dropped from ~9,274 stale
trades to ~6,900 on the current tree before the fix).

Usage:
  PYTHONPATH=. .venv/bin/python -m strategies.multi_ticker_swing.backtest.sweep_v2_clean --top-n 100
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from strategies.multi_ticker_swing.backtest.sweep_v2 import (
    CONFIRM_MAX_BARS_5M,
    ENTRY_THRESHOLDS,
    EXIT_STRATEGIES,
    TickerData,
    compute_grouped_metrics,
    compute_metrics,
    load_proba,
    load_raw_30m,
    load_raw_5m,
    select_top_tickers,
    simulate_ticker,
)
from strategies.multi_ticker_swing.config.pipeline_config import BACKTEST_RESULTS_DIR, TRADING_BLACKLIST

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

OUT_DIR = BACKTEST_RESULTS_DIR / "sweep_v2_clean"


def build_ticker_data(tickers: list[str], proba: pd.DataFrame) -> dict[str, TickerData]:
    out: dict[str, TickerData] = {}
    for t in tickers:
        try:
            raw_30m = load_raw_30m(t)
            raw_5m = load_raw_5m(t)
            proba_t = proba[proba["ticker"] == t]
            td = TickerData(t, raw_30m, raw_5m, proba_t)
            if td.has_5m:
                out[t] = td
        except Exception as exc:
            logger.warning("Skipping %s: %s", t, exc)
    return out


def sweep_split(ticker_data: dict[str, TickerData]) -> pd.DataFrame:
    results = []
    total = len(ENTRY_THRESHOLDS) * len(CONFIRM_MAX_BARS_5M) * len(EXIT_STRATEGIES)
    i = 0
    for entry_thresh in ENTRY_THRESHOLDS:
        for confirm_bars in CONFIRM_MAX_BARS_5M:
            for exit_cfg in EXIT_STRATEGIES:
                i += 1
                rows = []
                for td in ticker_data.values():
                    rows.extend(simulate_ticker(td, entry_thresh, confirm_bars, exit_cfg, trend_filter=False))
                trades_df = pd.DataFrame(rows, columns=[
                    "ticker", "direction", "signal_idx", "exit_idx",
                    "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars",
                ]) if rows else pd.DataFrame()
                m = compute_metrics(trades_df)
                m.update(combo_name=f"e{entry_thresh}_c{confirm_bars}_{exit_cfg['name']}",
                         entry_threshold=entry_thresh, confirm_5m_bars=confirm_bars,
                         exit_strategy=exit_cfg["name"])
                results.append(m)
                if i % 40 == 0:
                    logger.info("(%d/%d) combos swept", i, total)
    return pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)


def run(top_n: int = 100) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_proba = load_proba("val")
    test_proba = load_proba("test")

    tickers = select_top_tickers(val_proba, top_n)
    tickers = [t for t in tickers if t not in TRADING_BLACKLIST]
    logger.info("Selected %d tickers (post-blacklist)", len(tickers))

    logger.info("Building VAL ticker data...")
    val_td = build_ticker_data(tickers, val_proba)
    logger.info("Building TEST ticker data...")
    test_td = build_ticker_data(tickers, test_proba)
    logger.info("VAL tickers=%d  TEST tickers=%d", len(val_td), len(test_td))

    t0 = time.time()
    logger.info("Sweeping %d combos on VAL...", len(ENTRY_THRESHOLDS) * len(CONFIRM_MAX_BARS_5M) * len(EXIT_STRATEGIES))
    val_results = sweep_split(val_td)
    val_results.to_csv(OUT_DIR / "val_sweep_summary.csv", index=False)
    logger.info("VAL sweep done in %.1fs", time.time() - t0)

    best = val_results.iloc[0]
    best_exit_cfg = next(e for e in EXIT_STRATEGIES if e["name"] == best["exit_strategy"])
    logger.info("VAL-picked combo: %s (val sharpe=%.3f, val n_trades=%d)",
               best["combo_name"], best["sharpe"], best["n_trades"])

    logger.info("Freezing VAL-picked combo on TEST...")
    rows = []
    for td in test_td.values():
        rows.extend(simulate_ticker(td, best["entry_threshold"], int(best["confirm_5m_bars"]),
                                    best_exit_cfg, trend_filter=False))
    frozen_trades = pd.DataFrame(rows, columns=[
        "ticker", "direction", "signal_idx", "exit_idx",
        "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars",
    ])
    frozen_trades.to_parquet(OUT_DIR / "best_v2_clean_trades.parquet", index=False)
    frozen_metrics = compute_metrics(frozen_trades)
    frozen_grouped = compute_grouped_metrics(frozen_trades) if not frozen_trades.empty else {}

    summary = {
        "method": "combo selected on VAL split by Sharpe, frozen and reported on TEST split",
        "n_tickers": len(tickers),
        "combo": {
            "name": best["combo_name"], "entry_threshold": float(best["entry_threshold"]),
            "confirm_5m_bars": int(best["confirm_5m_bars"]), "exit_strategy": best["exit_strategy"],
        },
        "val_selection_metrics": {k: best[k] for k in
                                  ("n_trades", "win_rate", "sharpe", "profit_factor", "long_wr", "short_wr")},
        "frozen_test_metrics": frozen_metrics,
        "frozen_test_grouped_direction": {k: v for k, v in frozen_grouped.items() if k.startswith("dir:")},
    }
    (OUT_DIR / "best_v2_clean_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if frozen_grouped:
        (OUT_DIR / "best_v2_clean_grouped.json").write_text(json.dumps(frozen_grouped, indent=2, default=str))
    logger.info("DONE -> %s", OUT_DIR)
    logger.info("Frozen TEST result: %s", json.dumps(frozen_metrics, indent=2, default=str))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=100)
    args = p.parse_args()
    run(top_n=args.top_n)
