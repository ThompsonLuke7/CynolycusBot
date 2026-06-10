"""Daily metadata snapshot builder.

This module accepts vendor-enriched CSV/parquet input for float, shares,
market cap, sector, industry, and short interest. Polygon reference support can
be added later without changing downstream schemas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategies.momentum_scalper.configs.settings import METADATA_DIR, ensure_data_dirs
from strategies.momentum_scalper.utils.io import clean_ticker, write_parquet


FIELDS = ["date", "ticker", "float", "shares_outstanding", "market_cap", "sector", "industry", "short_interest"]


def build_metadata_snapshot(source: Path, snapshot_date: str) -> pd.DataFrame:
    raw = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if raw.empty:
        return pd.DataFrame(columns=FIELDS)
    lower = {c.lower().strip(): c for c in raw.columns}
    ticker_col = lower.get("ticker") or lower.get("symbol") or raw.columns[0]
    out = pd.DataFrame({"date": pd.Timestamp(snapshot_date).strftime("%Y-%m-%d"), "ticker": raw[ticker_col].map(clean_ticker)})
    for target in FIELDS[2:]:
        source_col = lower.get(target) or lower.get(target.replace("_", " "))
        out[target] = raw[source_col] if source_col else pd.NA
    for col in ["float", "shares_outstanding", "market_cap", "short_interest"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[FIELDS].drop_duplicates(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata daily snapshot")
    parser.add_argument("source", type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", type=Path, default=METADATA_DIR)
    args = parser.parse_args()
    ensure_data_dirs()
    df = build_metadata_snapshot(args.source, args.date)
    path = args.output_dir / f"{pd.Timestamp(args.date):%Y-%m-%d}.parquet"
    write_parquet(df, path)
    print(f"wrote {len(df):,} metadata rows to {path}")


if __name__ == "__main__":
    main()
