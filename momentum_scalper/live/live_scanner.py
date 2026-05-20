"""Live top-runner scanner.

The scanner accepts a callable data provider so it can be connected to Polygon,
Alpaca, or an in-memory replay feed without changing ranking code.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pandas as pd

from momentum_scalper.scanners.historical_scanner import reconstruct_premarket_scanner


BarProvider = Callable[[], pd.DataFrame]


def scan_once(provider: BarProvider, top_n: int = 20) -> pd.DataFrame:
    bars = provider()
    snapshot = reconstruct_premarket_scanner(bars)
    if snapshot.empty:
        return snapshot
    latest_ts = snapshot["timestamp"].max()
    return snapshot[snapshot["timestamp"].eq(latest_ts)].nsmallest(top_n, "scanner_rank").reset_index(drop=True)


def scan_loop(provider: BarProvider, interval_seconds: int = 15, top_n: int = 20) -> Iterator[pd.DataFrame]:
    interval_seconds = max(5, min(60, int(interval_seconds)))
    while True:
        yield scan_once(provider, top_n=top_n)
        time.sleep(interval_seconds)
