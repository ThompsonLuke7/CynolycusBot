from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.API.Alpaca_API.core.config import AlpacaConfig
from strategies.multi_ticker_swing.config.pipeline_config import RAW_30M_DIR, RAW_5M_DIR, TRADING_BLACKLIST
from strategies.multi_ticker_swing.data.fetch_data import universe_tickers


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i:i + size]


def load_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        tickers = [str(ticker).strip().upper() for ticker in args.tickers]
    elif args.trading_universe_json:
        data = json.loads(Path(args.trading_universe_json).read_text())
        tiers = {int(x) for x in args.tiers}
        tickers = [
            str(ticker).upper()
            for ticker, cfg in data.items()
            if int(cfg.get("tier", 0) or 0) in tiers
        ]
    else:
        tickers = universe_tickers(args.universe)
    blacklist = {str(t).upper() for t in TRADING_BLACKLIST}
    return sorted({ticker for ticker in tickers if ticker and ticker not in blacklist})


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "timestamp" not in out.columns and out.index.name and str(out.index.name).lower() == "timestamp":
        out = out.reset_index()
        out.columns = [str(c).lower() for c in out.columns]
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    ny = out["timestamp"].dt.tz_convert("America/New_York")
    minutes = ny.dt.hour * 60 + ny.dt.minute
    out = out.loc[(minutes >= 570) & (minutes <= 960)].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def latest_timestamp(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        df = normalise_frame(pd.read_parquet(path))
    except Exception:
        return None
    if df.empty:
        return None
    return pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()


def save_symbol_frame(symbol: str, fetched: pd.DataFrame, raw_dir: Path) -> bool:
    if fetched.empty:
        return False
    out = fetched[fetched["symbol"].astype(str).str.upper() == symbol].copy()
    if out.empty:
        return False
    out = normalise_frame(out)
    path = raw_dir / f"{symbol}.parquet"
    if path.exists():
        try:
            old = normalise_frame(pd.read_parquet(path))
            out = pd.concat([old, out], ignore_index=True)
        except Exception as exc:
            logging.warning("[%s] could not merge old cache, replacing file: %s", symbol, exc)
    out = (
        out.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-fetch and merge multi-ticker swing 5m execution bars.")
    parser.add_argument("--universe", default="Data/shared/universe/shared_universe.csv")
    parser.add_argument("--trading-universe-json", default="strategies/multi_ticker_swing/config/trading_universe.json")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--tiers", nargs="+", default=["1", "2"])
    parser.add_argument("--start", default="2026-05-01T13:30:00Z")
    parser.add_argument("--end", default="2026-06-05T21:00:00Z")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=500_000)
    parser.add_argument("--feed", default="IEX", choices=["IEX", "SIP"])
    parser.add_argument("--bar-minutes", type=int, default=5, choices=[5, 30])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tickers = load_tickers(args)
    start = parse_time(args.start)
    end = parse_time(args.end)
    stale_before = pd.Timestamp(start)
    raw_dir = RAW_5M_DIR if args.bar_minutes == 5 else RAW_30M_DIR
    requested = [
        ticker
        for ticker in tickers
        if args.force or (latest_timestamp(raw_dir / f"{ticker}.parquet") or pd.Timestamp.min.tz_localize("UTC")) < stale_before
    ]
    logging.info(
        "tickers=%d requested=%d chunk_size=%d feed=%s bar_minutes=%d out=%s",
        len(tickers),
        len(requested),
        args.chunk_size,
        args.feed,
        args.bar_minutes,
        raw_dir,
    )
    if not requested:
        return 0

    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(api_key=cfg.key_id, secret_key=cfg.secret_key)
    timeframe = TimeFrame(args.bar_minutes, TimeFrameUnit.Minute)
    feed = DataFeed.IEX if args.feed.upper() == "IEX" else DataFeed.SIP

    saved = 0
    for ci, batch in enumerate(chunks(requested, args.chunk_size), 1):
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
                feed=feed,
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
            saved += int(save_symbol_frame(symbol, df, raw_dir))
        logging.info("batch %d complete cumulative_saved=%d", ci, saved)
    logging.info("complete cumulative_saved=%d requested=%d", saved, len(requested))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
