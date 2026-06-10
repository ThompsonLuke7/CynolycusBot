from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from core.API.Alpaca_API.market_data.fetch_intraday import fetch_intraday


def _month_starts(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    starts: list[dt.datetime] = []
    cur = dt.datetime(start.year, start.month, 1, tzinfo=dt.timezone.utc)
    if cur < start:
        starts.append(start)
        if start.month == 12:
            cur = dt.datetime(start.year + 1, 1, 1, tzinfo=dt.timezone.utc)
        else:
            cur = dt.datetime(start.year, start.month + 1, 1, tzinfo=dt.timezone.utc)
    while cur < end:
        starts.append(cur)
        if cur.month == 12:
            cur = dt.datetime(cur.year + 1, 1, 1, tzinfo=dt.timezone.utc)
        else:
            cur = dt.datetime(cur.year, cur.month + 1, 1, tzinfo=dt.timezone.utc)
    starts.append(end)
    return starts


def main() -> None:
    start = dt.datetime(2020, 12, 1, 14, 30, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 4, 2, 21, 0, tzinfo=dt.timezone.utc)
    out_path = REPO_ROOT / "Data" / "raw" / "spy" / "spy_intraday_1min.parquet"
    tmp_dir = REPO_ROOT / "Data" / "raw" / "spy" / "_tmp_1min_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    boundaries = _month_starts(start, end)
    frames: list[pd.DataFrame] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        chunk_path = tmp_dir / f"spy_1min_{left:%Y%m%d}_{right:%Y%m%d}.parquet"
        if chunk_path.exists():
            df = pd.read_parquet(chunk_path)
        else:
            df = fetch_intraday(
                ticker="SPY",
                start=left,
                end=right,
                timeframe="1Min",
                limit=10000,
                feed="IEX",
                save_path=str(chunk_path),
                paginate=False,
            )
        print(f"{left.date()} -> {right.date()}: {len(df)} rows")
        if not df.empty:
            frames.append(df)

    if not frames:
        raise SystemExit("No rows fetched.")

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = (
        out.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"rows {len(out)}")
    print(f"range {out['timestamp'].min()} -> {out['timestamp'].max()}")


if __name__ == "__main__":
    main()
