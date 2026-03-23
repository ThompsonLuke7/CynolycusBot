from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from alpaca.data.enums import DataFeed

from API.Alpaca_API.market_data.fetch_intraday import fetch_intraday
from API.Alpaca_API.runners.live_runner import _normalize_prefill_1m_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append missing Alpaca 1m bars to a local prefill parquet/csv for faster live startup."
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol.")
    parser.add_argument(
        "--prefill-path",
        default="Data/raw/spy/spy_intraday_1min.parquet",
        help="Existing local 1m prefill file (.parquet or .csv).",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Optional output path. Defaults to overwrite --prefill-path.",
    )
    parser.add_argument("--feed", default="IEX", choices=["IEX", "SIP"], help="Alpaca feed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prefill_path = Path(args.prefill_path)
    if not prefill_path.exists():
        raise FileNotFoundError(f"Missing prefill file: {prefill_path}")

    if prefill_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(prefill_path)
    elif prefill_path.suffix.lower() == ".csv":
        df = pd.read_csv(prefill_path)
    else:
        raise ValueError("Prefill file must be .parquet or .csv")

    df = _normalize_prefill_1m_frame(df, symbol=args.symbol)
    latest_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna().max()
    if pd.isna(latest_ts):
        raise ValueError("Could not determine latest timestamp from prefill file.")

    feed = DataFeed.SIP if str(args.feed).strip().upper() == "SIP" else DataFeed.IEX
    fetch_start = latest_ts + pd.Timedelta(minutes=1)
    now_utc = pd.Timestamp.now(tz="UTC")
    if fetch_start >= now_utc:
        print(f"No update needed. Latest local bar is already {latest_ts}.")
        return

    fetched_df = fetch_intraday(
        ticker=args.symbol,
        start=fetch_start.isoformat(),
        timeframe="1Min",
        limit=100000,
        feed=feed,
        save_path=None,
    )
    if fetched_df is None or fetched_df.empty:
        print(f"No newer Alpaca 1m bars returned after {latest_ts}.")
        return

    fetched_df = _normalize_prefill_1m_frame(fetched_df, symbol=args.symbol)
    fetched_df = fetched_df[fetched_df["timestamp"] > latest_ts].copy()
    if fetched_df.empty:
        print("Fetched bars were all duplicates; nothing written.")
        return

    combined = pd.concat([df, fetched_df], axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp")
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last").sort_values("timestamp")

    out_path = Path(args.out_path) if args.out_path else prefill_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        combined.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".csv":
        combined.to_csv(out_path, index=False)
    else:
        raise ValueError("Output path must be .parquet or .csv")

    print(
        f"Updated prefill: wrote {len(combined):,} rows to {out_path} "
        f"(appended {len(fetched_df):,} rows from {fetched_df['timestamp'].min()} to {fetched_df['timestamp'].max()})."
    )


if __name__ == "__main__":
    main()
