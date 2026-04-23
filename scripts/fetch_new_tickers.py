"""
Fetch 30m and 5m data for newly added universe tickers.
Skips tickers that already have data files (cache-hit behavior).

Run:
  python -m scripts.fetch_new_tickers
"""
from __future__ import annotations

import logging
from multi_ticker_swing.data.fetch_data import fetch_ticker
from multi_ticker_swing.config.pipeline_config import (
    RAW_30M_DIR, RAW_5M_DIR, TRAIN_START, TRAIN_END,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

NEW_TICKERS = [
    # Already in CSV, data missing
    "OXY", "OIH",
    # New cross-context / regime ETFs
    "EEM", "FXI", "KRE", "ARKK", "DIA",
    # High-ATR momentum stocks
    "RIVN", "LCID", "ENPH", "FSLR", "PLUG", "SOUN",
    # Investment banks / financial cycle
    "GS", "MS", "BX",
    # Better energy E&P
    "EOG", "DVN", "MRO",
    # Leveraged ETFs for extreme ATR training
    "TQQQ", "SOXL", "LABU",
    # Semiconductor equipment
    "LRCX", "KLAC", "ON",
]


def main() -> None:
    logger.info("Fetching %d new tickers (30m + 5m)...", len(NEW_TICKERS))
    for i, ticker in enumerate(NEW_TICKERS, 1):
        logger.info("(%d/%d) %s", i, len(NEW_TICKERS), ticker)
        fetch_ticker(ticker, start=TRAIN_START, end=TRAIN_END,
                     alpaca_timeframe="30Min", base_dir=RAW_30M_DIR, force=False)
        fetch_ticker(ticker, start=TRAIN_START, end=TRAIN_END,
                     alpaca_timeframe="5Min",  base_dir=RAW_5M_DIR,  force=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
