"""Per-ticker static features exactly as the 30m ranker's training matrix holds them.

``training_ticker_meta.csv`` was extracted on 2026-11-23 from
``data/processed/training_matrix.parquet`` (the 2026-06 OOF ranker build, 1,133 tickers,
one value per ticker). The live builder used to re-derive these from
``swing_trader_universe_v3.csv`` (212 rows, most caps "Unknown"), which served
market_cap_bucket = 0 for every live name while training saw 0-5. sector_id is 0 for every
training row, so it is 0 here too.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

TRAINING_TICKER_META = Path(__file__).resolve().parent / "training_ticker_meta.csv"


@lru_cache(maxsize=1)
def _table() -> dict[str, dict[str, int]]:
    df = pd.read_csv(TRAINING_TICKER_META)
    return {str(r.ticker).upper(): {"sector_id": int(r.sector_id), "market_cap_bucket": int(r.market_cap_bucket)}
            for r in df.itertuples(index=False)}


def training_meta(ticker: str) -> dict[str, int] | None:
    """{"sector_id", "market_cap_bucket"} the training matrix used, or None if untrained."""
    return _table().get(str(ticker).upper())
