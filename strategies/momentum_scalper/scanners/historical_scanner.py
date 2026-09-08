"""Historical entry point for the causal momentum-scalper scanner."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategies.momentum_scalper.configs.v1 import ScannerConfigV1
from strategies.momentum_scalper.configs.settings import MINUTE_BARS_DIR, SCANNER_SNAPSHOTS_DIR, ensure_data_dirs
from strategies.momentum_scalper.scanners.premarket import reconstruct_premarket_snapshot
from strategies.momentum_scalper.utils.io import add_session_columns, write_parquet


def load_bars_for_day(day: str, bars_dir: Path = MINUTE_BARS_DIR) -> pd.DataFrame:
    """Load the selected day only; labels use this bounded helper."""

    target = pd.Timestamp(day).strftime("%Y-%m-%d")
    frames: list[pd.DataFrame] = []
    for path in bars_dir.glob("ticker=*/*.parquet"):
        frame = add_session_columns(pd.read_parquet(path))
        part = frame[frame["date"] == target]
        if not part.empty:
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_bars_through_day(
    day: str,
    *,
    bars_dir: Path = MINUTE_BARS_DIR,
    lookback_calendar_days: int = 45,
) -> pd.DataFrame:
    """Load current and prior sessions required for causal close/RVOL context."""

    end = pd.Timestamp(day).normalize()
    start = end - pd.Timedelta(days=lookback_calendar_days)
    frames: list[pd.DataFrame] = []
    for path in bars_dir.glob("ticker=*/*.parquet"):
        frame = add_session_columns(pd.read_parquet(path))
        part = frame[(frame["date"] >= start.strftime("%Y-%m-%d")) & (frame["date"] <= end.strftime("%Y-%m-%d"))]
        if not part.empty:
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_optional(path: Path | None) -> pd.DataFrame:
    return pd.read_parquet(path) if path is not None and path.exists() else pd.DataFrame()


def build_daily_snapshot(
    day: str,
    *,
    metadata_path: Path | None = None,
    news_path: Path | None = None,
    quotes_path: Path | None = None,
    halts_path: Path | None = None,
    config: ScannerConfigV1 = ScannerConfigV1(),
    output_dir: Path = SCANNER_SNAPSHOTS_DIR,
) -> Path:
    """Build one final premarket snapshot without fabricating missing inputs."""

    ensure_data_dirs()
    bars = load_bars_through_day(day)
    decision = pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} 09:29", tz="America/New_York")
    snapshot = reconstruct_premarket_snapshot(
        bars,
        decision_at=decision,
        metadata=_read_optional(metadata_path),
        news=_read_optional(news_path),
        quotes=_read_optional(quotes_path),
        halts=_read_optional(halts_path),
        config=config,
    )
    return write_parquet(snapshot, output_dir / f"{pd.Timestamp(day):%Y-%m-%d}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal momentum-scalper premarket snapshot")
    parser.add_argument("--day", required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--quotes", type=Path)
    parser.add_argument("--halts", type=Path)
    args = parser.parse_args()
    path = build_daily_snapshot(
        args.day, metadata_path=args.metadata, news_path=args.news,
        quotes_path=args.quotes, halts_path=args.halts,
    )
    print(f"wrote causal scanner snapshot to {path}")


if __name__ == "__main__":
    main()


__all__ = ["build_daily_snapshot", "load_bars_for_day", "load_bars_through_day", "reconstruct_premarket_snapshot"]
