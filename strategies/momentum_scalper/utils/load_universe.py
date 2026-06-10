"""Load and filter a US equity universe.

The default path is offline-friendly: pass NASDAQ Trader-style CSV files with
symbol columns, or a previously enriched parquet/CSV with price and market_cap.
When no source files are provided, the loader writes an empty schema so the rest
of the pipeline can be wired before data access is configured.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategies.momentum_scalper.configs.settings import ALL_EQUITIES_PATH, ensure_data_dirs
from strategies.momentum_scalper.utils.io import clean_ticker, write_parquet


EXCHANGES = ("NASDAQ", "NYSE", "AMEX")
SCHEMA = ["ticker", "exchange", "name", "price", "market_cap"]


def _standardize_frame(df: pd.DataFrame, exchange: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SCHEMA)
    columns = {c.lower().strip(): c for c in df.columns}
    ticker_col = columns.get("ticker") or columns.get("symbol") or df.columns[0]
    name_col = columns.get("name") or columns.get("security name") or columns.get("company")
    exchange_col = columns.get("exchange") or columns.get("market")
    price_col = columns.get("price") or columns.get("lastsale") or columns.get("last_sale")
    cap_col = columns.get("market_cap") or columns.get("marketcap") or columns.get("market cap")

    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].map(clean_ticker)
    out["exchange"] = df[exchange_col].astype(str) if exchange_col else (exchange or "UNKNOWN")
    out["name"] = df[name_col].astype(str) if name_col else ""
    out["price"] = pd.to_numeric(df[price_col].astype(str).str.replace("$", "", regex=False), errors="coerce") if price_col else pd.NA
    out["market_cap"] = pd.to_numeric(df[cap_col], errors="coerce") if cap_col else pd.NA
    out = out[out["ticker"].ne("")]
    return out[SCHEMA].drop_duplicates("ticker")


def load_universe(
    source_paths: list[Path] | None = None,
    max_price: float = 50.0,
    max_market_cap: float = 10_000_000_000.0,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in source_paths or []:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(path, sep="\t" if suffix == ".txt" else None, engine="python")
        exchange = next((ex for ex in EXCHANGES if ex.lower() in path.name.lower()), None)
        frames.append(_standardize_frame(raw, exchange))

    universe = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)
    if universe.empty:
        return universe
    price_ok = universe["price"].isna() | (universe["price"] < max_price)
    cap_ok = universe["market_cap"].isna() | (universe["market_cap"] < max_market_cap)
    return universe[price_ok & cap_ok].sort_values(["exchange", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all_equities.parquet")
    parser.add_argument("sources", nargs="*", type=Path, help="CSV/TXT/parquet universe source files")
    parser.add_argument("--output", type=Path, default=ALL_EQUITIES_PATH)
    args = parser.parse_args()
    ensure_data_dirs()
    df = load_universe(args.sources)
    write_parquet(df, args.output)
    print(f"wrote {len(df):,} tickers to {args.output}")


if __name__ == "__main__":
    main()
