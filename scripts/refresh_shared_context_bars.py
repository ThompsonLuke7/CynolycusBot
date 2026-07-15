"""Refresh and validate live 4H context caches.

The base 4H modules read SPY/QQQ/IWM/VIXY/sector context from
``Data/shared/bars/4h``.  Those files are distinct from the ticker universe,
so this explicit step is required alongside the full-universe bar catch-up.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategies.momentum_expansion.config.momentum_config import CONTEXT_TICKERS, SECTOR_ETFS
from strategies.momentum_expansion.data.bars import fetch_context_bars


REPO = Path(__file__).resolve().parents[1]
BARS_4H = REPO / "Data/shared/bars/4h"


def _latest_bar(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["timestamp"])
    if df.empty:
        return None
    return pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()


def refresh_and_validate(*, max_stale_days: float = 5.0) -> dict[str, int]:
    counts = fetch_context_bars()
    now = pd.Timestamp.now(tz="UTC")
    stale: list[str] = []
    for symbol in list(CONTEXT_TICKERS) + list(SECTOR_ETFS):
        latest = _latest_bar(BARS_4H / f"{symbol}.parquet")
        age_days = float("inf") if latest is None else (now - latest).total_seconds() / 86_400.0
        if age_days > max_stale_days:
            stale.append(f"{symbol}={latest} ({age_days:.1f}d)")
    if stale:
        raise RuntimeError("stale live context 4H cache(s): " + ", ".join(stale))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh and validate live shared context bars.")
    parser.add_argument("--max-stale-days", type=float, default=5.0)
    args = parser.parse_args()
    print(refresh_and_validate(max_stale_days=args.max_stale_days))


if __name__ == "__main__":
    main()
