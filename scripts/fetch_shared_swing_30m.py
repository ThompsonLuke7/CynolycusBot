from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.multi_ticker_swing.config.pipeline_config import RAW_30M_DIR, TRAIN_END, TRAIN_START
from strategies.multi_ticker_swing.data.fetch_data import fetch_ticker, universe_tickers


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch missing 30m bars for a swing universe.")
    parser.add_argument("--universe", default="Data/shared/universe/shared_universe.csv")
    parser.add_argument("--start", default=TRAIN_START)
    parser.add_argument("--end", default=TRAIN_END)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tickers = universe_tickers(args.universe)
    missing = [
        ticker for ticker in tickers
        if args.force or not (RAW_30M_DIR / f"{ticker}.parquet").exists()
    ]
    logging.info("tickers=%d missing=%d workers=%d", len(tickers), len(missing), args.workers)

    def fetch_one(ticker: str) -> tuple[str, bool]:
        df = fetch_ticker(
            ticker,
            start=args.start,
            end=args.end,
            alpaca_timeframe="30Min",
            base_dir=RAW_30M_DIR,
            force=args.force,
        )
        return ticker, df is not None

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_one, ticker): ticker for ticker in missing}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker = futures[fut]
            try:
                _, saved = fut.result()
                ok += int(saved)
                logging.info("progress %d/%d saved=%d latest=%s", i, len(missing), ok, ticker)
            except Exception as exc:
                logging.exception("[%s] failed: %s", ticker, exc)
    logging.info("complete saved=%d missing_requested=%d", ok, len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
