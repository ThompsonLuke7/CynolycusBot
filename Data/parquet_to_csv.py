import argparse
from pathlib import Path
import sys

import pandas as pd


def _pick_time_series(df: pd.DataFrame) -> pd.Series | None:
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    for col in ("timestamp", "date", "datetime", "time"):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    return None


def _filter_by_time(
    df: pd.DataFrame, start: str | None, end: str | None
) -> pd.DataFrame:
    if not start and not end:
        return df
    series = _pick_time_series(df)
    if series is None:
        print("No datetime index or timestamp/date column found; skipping time filter.")
        return df
    mask = pd.Series(True, index=df.index)
    if start:
        start_dt = pd.to_datetime(start, errors="coerce")
        if pd.isna(start_dt):
            raise ValueError(f"Invalid --start datetime: {start}")
        mask &= series >= start_dt
    if end:
        end_dt = pd.to_datetime(end, errors="coerce")
        if pd.isna(end_dt):
            raise ValueError(f"Invalid --end datetime: {end}")
        mask &= series <= end_dt
    return df.loc[mask]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a parquet file as CSV.")
    parser.add_argument("path", help="Path to a parquet file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Write CSV to this path (defaults to the parquet path with .csv).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Rows to output (0 or negative for all).",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Output tail instead of head.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated column list to include.",
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Start datetime filter."
    )
    parser.add_argument("--end", type=str, default=None, help="End datetime filter.")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Omit the index column in CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    df = pd.read_parquet(path)

    if args.columns:
        requested = [c.strip() for c in args.columns.split(",") if c.strip()]
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns: {', '.join(missing)}")
        df = df[requested]

    df = _filter_by_time(df, args.start, args.end)
    ts_series = _pick_time_series(df)
    if ts_series is not None:
        ts = pd.to_datetime(ts_series, errors="coerce", utc=True)
        ts = ts.dt.tz_convert("America/New_York")
        if "timestamp" in df.columns:
            df["timestamp"] = ts
        else:
            df.insert(0, "timestamp", ts)
    else:
        print("No timestamp/date column or datetime index found; skipping time zone conversion.")
    rows = int(args.rows)
    if rows > 0:
        df = df.tail(rows) if args.tail else df.head(rows)

    csv_text = df.to_csv(index=not args.no_index)
    output_path = Path(args.output) if args.output else path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_text, encoding="utf-8")
    print(f"Wrote CSV to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
