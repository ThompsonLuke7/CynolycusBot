from __future__ import annotations

import datetime as dt
import argparse
import os
from typing import Optional

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .config import AlpacaConfig
from Data.load_data import get_ticker_raw_dir
from Data.retrieve_data import normalize_ticker


def _parse_time(ts: dt.datetime | str) -> dt.datetime:
    """Ensure timezone-aware UTC datetime."""
    if isinstance(ts, dt.datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc)
    # assume ISO string; handle trailing Z
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
        dt.timezone.utc
    )


def _to_timeframe(timeframe: str) -> TimeFrame:
    tf_val = timeframe.lower().rstrip("s")
    if tf_val.endswith("min"):
        minutes = int(tf_val.replace("min", "") or "1")
        return TimeFrame(minutes, TimeFrameUnit.Minute)
    if tf_val.endswith("hour"):
        hours = int(tf_val.replace("hour", "") or "1")
        return TimeFrame(hours, TimeFrameUnit.Hour)
    if tf_val.endswith("day"):
        days = int(tf_val.replace("day", "") or "1")
        return TimeFrame(days, TimeFrameUnit.Day)
    if tf_val.endswith("d"):
        days = int(tf_val.replace("d", "") or "1")
        return TimeFrame(days, TimeFrameUnit.Day)
    raise ValueError(
        f"Unsupported timeframe '{timeframe}'. Use e.g. 1Min, 5Min, 15Min, 1Hour, 1Day."
    )


def _timeframe_slug(timeframe: str) -> str:
    tf_val = timeframe.lower().rstrip("s")
    if tf_val.endswith("min"):
        minutes = int(tf_val.replace("min", "") or "1")
        return f"{minutes}min"
    if tf_val.endswith("hour"):
        hours = int(tf_val.replace("hour", "") or "1")
        return f"{hours}hr"
    if tf_val.endswith("day"):
        days = int(tf_val.replace("day", "") or "1")
        return f"{days}d"
    if tf_val.endswith("d"):
        days = int(tf_val.replace("d", "") or "1")
        return f"{days}d"
    return tf_val


def fetch_intraday(
    *,
    ticker: str,
    start: dt.datetime | str,
    end: dt.datetime | str | None = None,
    timeframe: str = "1Min",
    limit: int = 100000,
    adjustment: str = "raw",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch intraday bars using Alpaca's official SDK (alpaca-py).

    Args:
        ticker: symbol to fetch (e.g., "AAPL", "$SPY").
        start: ISO string or datetime for the beginning (inclusive).
        end: ISO string or datetime for the end (exclusive). If None, defaults to now.
        timeframe: e.g., "1Min", "5Min", "15Min", "1Hour", "1Day".
        limit: maximum bars to request (SDK handles pagination internally).
        adjustment: "raw", "split", or "all".
        save_path: optional path (csv/parquet) to persist results.
    """
    clean_ticker = normalize_ticker(ticker)
    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(
        api_key=cfg.key_id,
        secret_key=cfg.secret_key,
    )

    tf = _to_timeframe(timeframe)
    start_dt = _parse_time(start)
    end_dt = _parse_time(end) if end is not None else dt.datetime.now(dt.timezone.utc)

    request = StockBarsRequest(
        symbol_or_symbols=clean_ticker,
        timeframe=tf,
        start=start_dt,
        end=end_dt,
        limit=limit,
        adjustment=Adjustment(adjustment),
        feed=DataFeed.IEX,
    )

    bars = client.get_stock_bars(request)
    df = bars.df
    if df is None or df.empty:
        return pd.DataFrame()

    # For single symbol the index is a MultiIndex; flatten to plain DataFrame.
    df = df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == clean_ticker]
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        ny = ts.dt.tz_convert("America/New_York")
        minutes = ny.dt.hour * 60 + ny.dt.minute
        regular_mask = ts.notna() & (minutes >= 570) & (minutes <= 960)
        df = df.loc[regular_mask].reset_index(drop=True)

    if save_path is None:
        slug = clean_ticker.lower()
        raw_dir = get_ticker_raw_dir(clean_ticker)
        raw_dir.mkdir(parents=True, exist_ok=True)
        tf_slug = _timeframe_slug(timeframe)
        save_path = str(raw_dir / f"{slug}_intraday_{tf_slug}.parquet")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if save_path.lower().endswith(".csv"):
            df.to_csv(save_path, index=False)
        else:
            df.to_parquet(save_path, index=False)

    return df


def fetch_intraday_spy(
    *,
    start: dt.datetime | str,
    end: dt.datetime | str | None = None,
    timeframe: str = "1Min",
    limit: int = 10000,
    adjustment: str = "raw",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    return fetch_intraday(
        ticker="SPY",
        start=start,
        end=end,
        timeframe=timeframe,
        limit=limit,
        adjustment=adjustment,
        save_path=save_path,
    )


def fetch_latest_quote(
    *,
    ticker: str = "SPY",
    feed: DataFeed = DataFeed.IEX,
    as_dataframe: bool = True,
) -> pd.DataFrame | dict:
    """
    Fetch the latest quote for a ticker (bid/ask).
    """
    clean_ticker = normalize_ticker(ticker)
    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(
        api_key=cfg.key_id,
        secret_key=cfg.secret_key,
    )
    request = StockLatestQuoteRequest(symbol_or_symbols=clean_ticker, feed=feed)
    resp = client.get_stock_latest_quote(request)

    quote = None
    if isinstance(resp, dict):
        quote = resp.get(clean_ticker) or resp.get(clean_ticker.upper())
    else:
        quote = resp

    if quote is None:
        raise ValueError("No quote returned from Alpaca.")

    payload = {
        "symbol": getattr(quote, "symbol", clean_ticker),
        "timestamp": getattr(quote, "timestamp", None),
        "bid_price": getattr(quote, "bid_price", None),
        "ask_price": getattr(quote, "ask_price", None),
        "bid_size": getattr(quote, "bid_size", None),
        "ask_size": getattr(quote, "ask_size", None),
        "bid_exchange": getattr(quote, "bid_exchange", None),
        "ask_exchange": getattr(quote, "ask_exchange", None),
    }
    if not as_dataframe:
        return payload
    return pd.DataFrame([payload])


def fetch_latest_quote_spy(
    *,
    feed: DataFeed = DataFeed.IEX,
    as_dataframe: bool = True,
) -> pd.DataFrame | dict:
    return fetch_latest_quote(ticker="SPY", feed=feed, as_dataframe=as_dataframe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch intraday bars and save to Data/<ticker>/raw."
    )
    parser.add_argument("--ticker", type=str, default="$SPY")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1Hour")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--adjustment", type=str, default="raw")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    now_utc = dt.datetime.now(dt.timezone.utc)
    default_start = now_utc - dt.timedelta(days=200)
    start = args.start or default_start
    end = args.end

    df = fetch_intraday(
        ticker=args.ticker,
        start=start,
        end=end,
        timeframe=args.timeframe,
        limit=args.limit,
        adjustment=args.adjustment,
        save_path=args.save_path,
    )

    print(df.tail())
