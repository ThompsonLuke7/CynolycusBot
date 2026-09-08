"""Live scanner facade using the same causal implementation as historical replay."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from strategies.momentum_scalper.configs.v1 import ScannerConfigV1
from strategies.momentum_scalper.scanners.premarket import reconstruct_premarket_snapshot


def scan_once(
    *,
    bars: pd.DataFrame,
    timestamp: datetime,
    metadata: pd.DataFrame,
    news: pd.DataFrame,
    quotes: pd.DataFrame,
    halts: pd.DataFrame | None = None,
    config: ScannerConfigV1 = ScannerConfigV1(),
    top_n: int = 20,
) -> pd.DataFrame:
    snapshot = reconstruct_premarket_snapshot(
        bars, decision_at=timestamp, metadata=metadata, news=news, quotes=quotes,
        halts=halts, config=config,
    )
    return snapshot[snapshot["eligible"]].nsmallest(top_n, "scanner_rank").reset_index(drop=True)


__all__ = ["scan_once"]
