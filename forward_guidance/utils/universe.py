"""Universe and sector ETF helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forward_guidance.config import SECTOR_ETFS, UNIVERSE_CSV


def normalize_sector(value: object) -> str:
    return str(value or "").strip().lower().replace("&", "and").replace(" ", "_").replace("-", "_")


def sector_to_etf(sector: object) -> str | None:
    return SECTOR_ETFS.get(normalize_sector(sector))


def load_universe(path: Path | str = UNIVERSE_CSV) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "sector", "sector_etf"])
    df = pd.read_csv(p)
    if "ticker" not in df.columns:
        raise ValueError(f"Universe file must include ticker column: {p}")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    if "sector_etf" not in df.columns:
        if "sector" in df.columns:
            df["sector_etf"] = df["sector"].map(sector_to_etf)
        else:
            df["sector_etf"] = None
    return df


def ticker_sector_etf(ticker: str, universe: pd.DataFrame | None = None) -> str | None:
    df = load_universe() if universe is None else universe
    if df.empty or "ticker" not in df.columns:
        return None
    rows = df.loc[df["ticker"].astype(str).str.upper() == str(ticker).upper()]
    if rows.empty:
        return None
    value = rows.iloc[0].get("sector_etf")
    if pd.isna(value):
        return None
    return str(value).upper()
