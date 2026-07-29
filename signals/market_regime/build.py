"""CLI to build the daily market-regime and sector-state tables.

Usage:
    python -m signals.market_regime.build [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Writes ``Data/shared/market_regime/daily_regime.parquet`` and
``sector_state.parquet`` atomically (temp file + rename, matching
strategies/intraday_structure/state_store.py's pattern).

``--start``/``--end`` filter the OUTPUT rows only; every rolling window is
always computed over the ticker's full cached history first, so trimming the
output never truncates a composite's lookback (which would reintroduce
warmup NaNs that shouldn't be there).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import DAILY_REGIME_PATH, SECTOR_STATE_PATH
from .daily_regime import build_daily_regime
from .sector_state import build_sector_state
from .timeutil import atomic_write_parquet

logger = logging.getLogger(__name__)


def _filter_dates(df: pd.DataFrame, *, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="Inclusive output start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Inclusive output end date (YYYY-MM-DD)")
    parser.add_argument("--out-regime", default=str(DAILY_REGIME_PATH), help="daily_regime.parquet output path")
    parser.add_argument("--out-sector-state", default=str(SECTOR_STATE_PATH), help="sector_state.parquet output path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Building daily_regime table...")
    regime = build_daily_regime()
    logger.info("Building sector_state table...")
    sector_state = build_sector_state()

    regime = _filter_dates(regime, start=args.start, end=args.end)
    sector_state = _filter_dates(sector_state, start=args.start, end=args.end)

    out_regime = Path(args.out_regime)
    out_sector_state = Path(args.out_sector_state)
    atomic_write_parquet(regime, out_regime, index=False)
    atomic_write_parquet(sector_state, out_sector_state, index=False)

    logger.info(
        "daily_regime: %d rows [%s .. %s] -> %s",
        len(regime),
        regime["date"].min() if len(regime) else None,
        regime["date"].max() if len(regime) else None,
        out_regime,
    )
    logger.info(
        "sector_state: %d rows [%s .. %s] -> %s",
        len(sector_state),
        sector_state["date"].min() if len(sector_state) else None,
        sector_state["date"].max() if len(sector_state) else None,
        out_sector_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
