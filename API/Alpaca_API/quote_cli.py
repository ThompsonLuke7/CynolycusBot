from __future__ import annotations

import argparse
import json

from alpaca.data.enums import DataFeed

from .fetch_intraday import fetch_latest_quote


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest quote from Alpaca.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--feed", default="IEX", choices=["IEX", "SIP"], help="IEX or SIP")
    parser.add_argument("--as-json", action="store_true", help="Print JSON instead of DataFrame")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feed = DataFeed.IEX if args.feed.upper() != "SIP" else DataFeed.SIP
    result = fetch_latest_quote(
        ticker=args.symbol,
        feed=feed,
        as_dataframe=not args.as_json,
    )
    if args.as_json:
        print(json.dumps(result, default=str, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
