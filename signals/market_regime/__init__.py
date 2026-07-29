"""Market-regime and sector-context signals (WS-A).

See docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md.
"""
from .daily_regime import build_daily_regime
from .sector_map import SECTOR_MAP, sector_etf_for
from .sector_state import build_sector_state

__all__ = [
    "build_daily_regime",
    "build_sector_state",
    "sector_etf_for",
    "SECTOR_MAP",
]
