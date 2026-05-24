"""CLI for scheduled event context features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from events.collectors import collect_earnings_dates, collect_macro_events, load_scheduled_events
from events.config import EVENT_FEATURES_PATH
from events.features import build_event_features


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scheduled event context pipeline.")
    parser.add_argument("--stage", choices=["collect-macro", "collect-earnings", "features"], required=True)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--timestamps-csv", default=None, help="CSV with timestamp,ticker rows for feature generation.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    if args.stage == "collect-macro":
        collect_macro_events(args.input_csv, output_path=output or None)
    elif args.stage == "collect-earnings":
        collect_earnings_dates(args.input_csv, output_path=output or None)
    else:
        if not args.timestamps_csv:
            raise ValueError("--timestamps-csv is required for --stage features")
        timestamps = _read_table(args.timestamps_csv)
        macro, earnings = load_scheduled_events()
        build_event_features(timestamps, macro, earnings, output_path=output or EVENT_FEATURES_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
