"""Configuration for signals.market_regime: required symbols, rolling windows,
output paths, and freshness tolerances.

Time-correctness contract (see
docs/superpowers/plans/2026-07-26-market-regime-and-sector-context.md §3):
every rolling statistic is trailing-only (no centered/smoothed windows),
z-scores use an explicit ``min_periods`` (NaN before that, never
zero-filled), and every output row carries both ``date`` (session date, ET,
tz-naive) and ``available_at`` (UTC timestamp at which the row became
knowable — that session's close plus a settle margin).
"""
from __future__ import annotations

from pathlib import Path

from strategies.momentum_expansion.config.momentum_config import SECTOR_ETFS

MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[1]
SHARED_DATA_ROOT = REPO_ROOT / "Data" / "shared"

MARKET_REGIME_DIR = SHARED_DATA_ROOT / "market_regime"
DAILY_REGIME_PATH = MARKET_REGIME_DIR / "daily_regime.parquet"
SECTOR_STATE_PATH = MARKET_REGIME_DIR / "sector_state.parquet"
SECTOR_ASSIGNMENTS_PATH = MARKET_REGIME_DIR / "sector_assignments.parquet"

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
# Sector ETFs reused verbatim from momentum_config (do not redefine).
SECTOR_ETFS_LIST: list[str] = list(SECTOR_ETFS)

# Non-sector-ETF tickers consumed directly by the daily_regime.py formulas
# (plan §4 WS-A). XLY/XLP are also sector ETFs, already covered by
# SECTOR_ETFS_LIST; listed again here only to document the formula inputs.
#
# CREDIT_DENOMINATOR: the plan's literal formula pairs HYG (high yield, ~3.5y
# effective duration) against LQD (investment grade, ~8-9y duration). That
# pairing is duration-contaminated, not a clean credit-quality signal — see
# the module docstring in daily_regime.py for the verified 2022 sign
# inversion. IEI (3-7y treasuries, ~4.5y duration — the closest duration
# match to HYG available as a cached ETF) is used as the denominator instead.
# LQD is still loaded so the original ratio can be exposed as a diagnostic
# column (credit_risk_hyg_lqd_z) — it must not feed any composite.
CREDIT_DENOMINATOR: str = "IEI"
CREDIT_DENOMINATOR_FALLBACK: str = "SHY"  # verified to also carry the correct (negative) 2022 sign

RISK_APPETITE_SYMBOLS: tuple[str, ...] = ("XLY", "XLP", "IWM", "SPY", "HYG", CREDIT_DENOMINATOR, "RSP")
LIQUIDITY_STRESS_SYMBOLS: tuple[str, ...] = ("SPY", "HYG", CREDIT_DENOMINATOR)
CREDIT_RISK_SYMBOLS: tuple[str, ...] = ("HYG", CREDIT_DENOMINATOR)

# Every ticker daily_regime.py / sector_state.py need cached daily bars for.
# LQD is retained only to compute the diagnostic credit_risk_hyg_lqd_z column.
REQUIRED_REGIME_SYMBOLS: tuple[str, ...] = tuple(
    sorted({"SPY", "IWM", "HYG", "LQD", "RSP", CREDIT_DENOMINATOR} | set(SECTOR_ETFS))
)

# Backfilled per plan defect D4 so the risk-appetite / short-history
# composites are meaningful over the full training window (RSP/SPY is one of
# the four risk_appetite_z legs). IEI was backfilled for the same reason once
# it was adopted as the credit-leg denominator (it originally only had 288
# rows spanning 2025-04-30..2026-06-23, and was stale on top of that). UUP
# and VIXY are backfilled alongside RSP because they share the same
# short-history defect, but neither is consumed by a WS-A formula yet: VIXY
# is already a separate 4H context ticker in strategies/momentum_expansion,
# and UUP is reserved for a possible future dollar-strength composite.
BACKFILL_SHORT_HISTORY_SYMBOLS: tuple[str, ...] = ("RSP", "UUP", "VIXY", CREDIT_DENOMINATOR)
BACKFILL_START = "2020-07-27"

# ---------------------------------------------------------------------------
# Rolling windows (plan §3: trailing-only, explicit min_periods)
# ---------------------------------------------------------------------------
ZSCORE_WINDOW = 252
ZSCORE_MIN_PERIODS = 252
COMPONENT_WINDOW_SHORT = 20     # Amihud_20 / DollarVolume_20 / RV20 / SMA20
COMPONENT_WINDOW_LONG = 50      # SMA50
EXCESS_RETURN_WINDOW = 21
EXCESS_RETURN_WINDOW_LONG = 63

# ---------------------------------------------------------------------------
# Time correctness
# ---------------------------------------------------------------------------
MARKET_CLOSE_ET = "16:00:00"
# Margin after the 16:00 ET close before the daily bar is treated as settled
# / finalized and safe to mark "available".
SETTLE_MARGIN_MINUTES = 30

# ---------------------------------------------------------------------------
# Freshness (consumed by build.py and any future live-gate consumer)
# ---------------------------------------------------------------------------
# The regime table's last session may lag "today" by at most this many
# sessions before a consumer should treat it as stale.
FRESHNESS_TOLERANCE_SESSIONS = 1

# ---------------------------------------------------------------------------
# Sector resolver (plan defects D1/D2) — empirical rolling-correlation
# fallback for tickers absent from the curated SECTOR_MAP. Ships OFF by
# default; only flips after ablation evidence (D2: silently reassigning
# tickers off the old XLK fallback changes rs_sector_5/20 for the deployed
# boosters, which were trained under XLK-fallback semantics).
# ---------------------------------------------------------------------------
SECTOR_RESOLVER_ENABLED = False
SECTOR_RESOLVER_WINDOW_DAYS = 120
SECTOR_RESOLVER_MIN_OBS = 60
SECTOR_RESOLVER_RECOMPUTE_FREQ = "MS"  # monthly recompute, point-in-time snapshot
