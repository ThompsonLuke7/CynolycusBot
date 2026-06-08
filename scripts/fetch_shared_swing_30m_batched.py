from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from API.Alpaca_API.core.config import AlpacaConfig
from multi_ticker_swing.config.pipeline_config import RAW_30M_DIR, TRAIN_END, TRAIN_START
from multi_ticker_swing.data.fetch_data import universe_tickers


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def chunks(values: list[str], size: int):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def save_symbol_frame(symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    out = df[df["symbol"].astype(str).str.upper() == symbol].copy()
    if out.empty:
        return
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    ny = ts.dt.tz_convert("America/New_York")
    minutes = ny.dt.hour * 60 + ny.dt.minute
    out = out.loc[ts.notna() & (minutes >= 570) & (minutes <= 960)].copy()
    out = (
        out.sort_values("timestamp")
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )
    RAW_30M_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RAW_30M_DIR / f"{symbol}.parquet", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-fetch missing shared swing 30m bars.")
    parser.add_argument("--universe", default="Data/shared/universe/shared_universe.csv")
    parser.add_argument("--start", default=TRAIN_START)
    parser.add_argument("--end", default=TRAIN_END)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=500_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tickers = universe_tickers(args.universe)
    missing = [
        ticker for ticker in tickers
        if args.force or not (RAW_30M_DIR / f"{ticker}.parquet").exists()
    ]
    logging.info("tickers=%d missing=%d chunk_size=%d", len(tickers), len(missing), args.chunk_size)
    if not missing:
        return 0

    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)
    timeframe = TimeFrame(30, TimeFrameUnit.Minute)
    start = parse_time(args.start)
    end = parse_time(args.end)

    saved = 0
    for ci, batch in enumerate(chunks(missing, args.chunk_size), 1):
        logging.info("batch %d symbols=%d first=%s last=%s", ci, len(batch), batch[0], batch[-1])
        page_token = None
        frames: list[pd.DataFrame] = []
        while True:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=args.limit,
                adjustment=Adjustment("split"),
                feed=DataFeed.IEX,
                page_token=page_token,
            )
            resp = client.get_stock_bars(req)
            if resp.df is not None and not resp.df.empty:
                frames.append(resp.df.reset_index())
            page_token = getattr(resp, "next_page_token", None)
            if not page_token:
                break
        if not frames:
            logging.warning("batch %d returned no bars", ci)
            continue
        df = pd.concat(frames, ignore_index=True)
        for symbol in batch:
            before = saved
            save_symbol_frame(symbol, df)
            if (RAW_30M_DIR / f"{symbol}.parquet").exists():
                saved += int(before == saved)
        logging.info("batch %d complete cumulative_saved=%d", ci, saved)
    logging.info("complete cumulative_saved=%d", saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
