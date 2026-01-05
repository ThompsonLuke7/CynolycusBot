from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .config import AlpacaConfig


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
        minutes = int(tf_val.replace("min", ""))
        return TimeFrame(minutes, TimeFrameUnit.Minute)
    if tf_val.endswith("hour"):
        hours = int(tf_val.replace("hour", ""))
        return TimeFrame(hours, TimeFrameUnit.Hour)
    raise ValueError(
        f"Unsupported timeframe '{timeframe}'. Use e.g. 1Min, 5Min, 15Min, 1Hour."
    )


def fetch_intraday_spy(
    *,
    start: dt.datetime | str,
    end: dt.datetime | str | None = None,
    timeframe: str = "1Min",
    limit: int = 10000,
    adjustment: str = "raw",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch intraday SPY bars using Alpaca's official SDK (alpaca-py).

    Args:
        start: ISO string or datetime for the beginning (inclusive).
        end: ISO string or datetime for the end (exclusive). If None, defaults to now.
        timeframe: e.g., "1Min", "5Min", "15Min", "1Hour".
        limit: maximum bars to request (SDK handles pagination internally).
        adjustment: "raw", "split", or "all".
        save_path: optional path (csv/parquet) to persist results.
    """
    cfg = AlpacaConfig.from_env()
    client = StockHistoricalDataClient(
        api_key=cfg.key_id,
        secret_key=cfg.secret_key,
    )

    tf = _to_timeframe(timeframe)
    start_dt = _parse_time(start)
    end_dt = _parse_time(end) if end is not None else dt.datetime.now(dt.timezone.utc)

    request = StockBarsRequest(
        symbol_or_symbols="SPY",
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
        df = df[df["symbol"] == "SPY"]
    df = df.sort_values("timestamp").reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if save_path.lower().endswith(".csv"):
            df.to_csv(save_path, index=False)
        else:
            df.to_parquet(save_path, index=False)

    return df


if __name__ == "__main__":
    now_utc = dt.datetime.now(dt.timezone.utc)
    default_start = now_utc - dt.timedelta(days=45)
    df = fetch_intraday_spy(
        start=default_start,
        timeframe="1hour",
        limit=10000,
        save_path=os.path.join("Data", "spy_intraday_1hr.parquet"),
    )

    print(df.tail())
