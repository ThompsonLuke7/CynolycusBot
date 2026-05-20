"""NASDAQ trading halt downloader."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from momentum_scalper.configs.settings import HALTS_DIR, ensure_data_dirs
from momentum_scalper.utils.io import clean_ticker, write_parquet


HALTS_URL = "https://www.nasdaqtrader.com/dynamic/symdir/tradehalts/tradehalts.txt"
FIELDS = ["halt_timestamp", "resume_timestamp", "ticker", "reason_code"]


def download_halts(url: str = HALTS_URL) -> pd.DataFrame:
    raw = pd.read_csv(url, sep="|")
    if raw.empty:
        return pd.DataFrame(columns=FIELDS)
    cols = {c.lower().strip(): c for c in raw.columns}
    symbol_col = cols.get("symbol") or cols.get("ticker") or raw.columns[0]
    reason_col = cols.get("reason code") or cols.get("reason") or cols.get("reason_code")
    halt_date_col = cols.get("halt date") or cols.get("date")
    halt_time_col = cols.get("halt time") or cols.get("time")
    resume_date_col = cols.get("resumption date") or halt_date_col
    resume_time_col = cols.get("resumption trade time") or cols.get("resumption quote time")

    out = pd.DataFrame()
    out["ticker"] = raw[symbol_col].map(clean_ticker)
    halt_dt = raw[halt_date_col].astype(str) + " " + raw[halt_time_col].astype(str) if halt_date_col and halt_time_col else pd.NA
    resume_dt = raw[resume_date_col].astype(str) + " " + raw[resume_time_col].astype(str) if resume_date_col and resume_time_col else pd.NA
    out["halt_timestamp"] = pd.to_datetime(halt_dt, utc=True, errors="coerce")
    out["resume_timestamp"] = pd.to_datetime(resume_dt, utc=True, errors="coerce")
    out["reason_code"] = raw[reason_col].astype(str) if reason_col else ""
    return out[FIELDS].dropna(subset=["halt_timestamp", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NASDAQ trading halts")
    parser.add_argument("--output", type=Path, default=HALTS_DIR / "halts.parquet")
    args = parser.parse_args()
    ensure_data_dirs()
    df = download_halts()
    write_parquet(df, args.output)
    print(f"wrote {len(df):,} halts to {args.output}")


if __name__ == "__main__":
    main()
