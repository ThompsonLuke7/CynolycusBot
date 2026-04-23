"""
Regenerate best_v2_trades.parquet, best_v2_grouped.json, best_v2_per_ticker.csv
for the confirmed best combo: e=0.70, confirm_5m=6, sl_atr=3.0, with TRADING_BLACKLIST applied.

Run:
  python -m scripts.regenerate_best_artifacts
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from multi_ticker_swing.backtest.sweep_v2 import (
    TickerData,
    compute_grouped_metrics,
    compute_metrics,
    load_proba,
    load_raw_30m,
    load_raw_5m,
    simulate_ticker,
    SWEEP_DIR,
)
from multi_ticker_swing.config.pipeline_config import (
    RAW_30M_DIR,
    TRADING_BLACKLIST,
    UNIVERSE_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

BEST_COMBO = {
    "entry_threshold": 0.70,
    "confirm_max_5m": 6,
    "exit_cfg": {
        "name": "trail_arm2.5_gb25_sl4.0",
        "arm_pct": 0.025,
        "giveback_pct": 0.25,
        "sl_atr": 4.0,
        "opp_exit": None,
        "max_days": None,
    },
}


def main() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading probabilities (split=test)...")
    proba = load_proba("test")

    all_tickers = proba["ticker"].unique().tolist()
    tickers = [t for t in all_tickers if (RAW_30M_DIR / f"{t}.parquet").exists()]
    blacklisted = [t for t in tickers if t in TRADING_BLACKLIST]
    tickers = [t for t in tickers if t not in TRADING_BLACKLIST]
    logger.info("Tickers: %d total, blacklist removed %d (%s)", len(tickers) + len(blacklisted), len(blacklisted), blacklisted)

    logger.info("Pre-building ticker data...")
    ticker_data: dict[str, TickerData] = {}
    for t in tickers:
        try:
            raw_30m = load_raw_30m(t)
            raw_5m = load_raw_5m(t)
            proba_t = proba[proba["ticker"] == t]
            td = TickerData(t, raw_30m, raw_5m, proba_t)
            if not td.has_5m:
                continue
            ticker_data[t] = td
        except Exception as e:
            logger.warning("Skipping %s: %s", t, e)

    logger.info("Built data for %d tickers", len(ticker_data))

    entry_thresh = BEST_COMBO["entry_threshold"]
    confirm_bars = BEST_COMBO["confirm_max_5m"]
    exit_cfg = BEST_COMBO["exit_cfg"]

    logger.info("Simulating best combo: e=%.2f, c=%d, %s ...", entry_thresh, confirm_bars, exit_cfg["name"])
    all_rows = []
    for td in ticker_data.values():
        trades = simulate_ticker(td, entry_thresh, confirm_bars, exit_cfg)
        all_rows.extend(trades)

    if not all_rows:
        logger.error("No trades generated — check data and probabilities.")
        return

    trades_df = pd.DataFrame(all_rows, columns=[
        "ticker", "direction", "signal_idx", "exit_idx",
        "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars",
    ])

    metrics = compute_metrics(trades_df)
    logger.info(
        "Best combo: n=%d  WR=%.1f%%  Sharpe=%+.2f  avgPnL=%+.3f%%  Hold=%.1fh  SL%%=%.0f  trail%%=%.0f",
        metrics["n_trades"], metrics["win_rate"] * 100, metrics["sharpe"],
        metrics["avg_pnl_pct"] * 100, metrics["avg_holding_bars"] * 0.5,
        metrics["exit_sl_pct"] * 100, metrics["exit_trail_pct"] * 100,
    )

    # Save trades
    trades_path = SWEEP_DIR / "best_v2_trades.parquet"
    trades_df.to_parquet(trades_path, index=False)
    logger.info("Saved %s", trades_path)

    # Grouped metrics
    grouped = compute_grouped_metrics(trades_df)
    grouped_path = SWEEP_DIR / "best_v2_grouped.json"
    with open(grouped_path, "w") as f:
        json.dump(grouped, f, indent=2, default=str)
    logger.info("Saved %s", grouped_path)

    # Per-ticker
    ticker_metrics = []
    for ticker, grp in trades_df.groupby("ticker"):
        m = compute_metrics(grp)
        m["ticker"] = ticker
        ticker_metrics.append(m)
    ticker_df = pd.DataFrame(ticker_metrics).sort_values("sharpe", ascending=False)
    per_ticker_path = SWEEP_DIR / "best_v2_per_ticker.csv"
    ticker_df.to_csv(per_ticker_path, index=False)
    logger.info("Saved %s", per_ticker_path)

    logger.info("\n=== TOP 15 TICKERS ===")
    for _, row in ticker_df.head(15).iterrows():
        logger.info("  %-6s  n=%3d  WR=%.1f%%  Sharpe=%+.2f  totalPnL=%+.2f%%",
                    row["ticker"], row["n_trades"], row["win_rate"] * 100,
                    row["sharpe"], row["total_pnl_pct"] * 100)

    logger.info("\n=== BOTTOM 10 TICKERS ===")
    for _, row in ticker_df.tail(10).iterrows():
        logger.info("  %-6s  n=%3d  WR=%.1f%%  Sharpe=%+.2f  totalPnL=%+.2f%%",
                    row["ticker"], row["n_trades"], row["win_rate"] * 100,
                    row["sharpe"], row["total_pnl_pct"] * 100)

    logger.info("\n=== SECTOR BREAKDOWN ===")
    for gn, gm in sorted(grouped.items()):
        if gm["n_trades"] >= 10:
            logger.info(
                "  %-40s  n=%4d  WR=%.1f%%  Sharpe=%+.2f  avgPnL=%+.3f%%",
                gn, gm["n_trades"], gm["win_rate"] * 100, gm["sharpe"], gm["avg_pnl_pct"] * 100,
            )


if __name__ == "__main__":
    main()
