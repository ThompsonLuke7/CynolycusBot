"""
Fetch US Treasury constant-maturity yields from FRED (free, no API key) and write
the parquet the Meta Ranker matrix expects.

Series: DGS3MO, DGS2, DGS10, DGS30 (daily, percent). FRED exposes a keyless CSV
download endpoint, so this needs no credentials.

Output columns: date, month3, year2, year10, year30, spread_2s10s, spread_3m10y, inverted
  -> signals/meta_context/data/processed/fmp_treasury_rates.parquet

  PYTHONPATH=. python scripts/fetch_treasury_rates.py
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "signals/meta_context/data/processed/fmp_treasury_rates.parquet"
SERIES = {"DGS3MO": "month3", "DGS2": "year2", "DGS10": "year10", "DGS30": "year30"}
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _fetch_series(sid: str) -> pd.Series:
    req = urllib.request.Request(FRED.format(sid=sid), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = sid.lower() if sid.lower() in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")  # FRED uses "." for holidays
    return df.set_index(date_col)[val_col].rename(SERIES[sid])


def main():
    cols = [_fetch_series(sid) for sid in SERIES]
    tr = pd.concat(cols, axis=1).sort_index()
    tr = tr.ffill().dropna(how="all")
    tr.index.name = "date"
    tr = tr.reset_index()
    tr["spread_2s10s"] = tr["year10"] - tr["year2"]
    tr["spread_3m10y"] = tr["year10"] - tr["month3"]
    tr["inverted"] = (tr["spread_3m10y"] < 0).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(OUT, index=False)
    print(f"wrote {len(tr):,} rows -> {OUT}")
    print(f"  range {tr['date'].min().date()} .. {tr['date'].max().date()}")
    print(tr.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
