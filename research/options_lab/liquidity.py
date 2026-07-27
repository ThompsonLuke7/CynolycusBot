"""Per-contract option liquidity/tradability gates from real cached Alpaca data.

Part of the options-instrument-routing experiment (Phase 1). Unlike
``core/option_liquidity.py`` (Schwab chain snapshot, live-only, TTL-cached),
this module reads from the historical parquet cache built by ``chain_cache``:
bar volume + trade count (``n``) from ``/v1beta1/options/bars`` and open
interest from ``/v2/options/contracts``.

Missing-data discipline mirrors ``core/option_liquidity.py``'s documented
lesson (see its module docstring): a contract whose OI or volume could not
be determined must resolve to ``None``, never to ``0``. ``is_tradable``
therefore fails closed on ``None`` — "unknown" is not "liquid enough" — but
it must never be produced by silently coercing a missing lookup to zero.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from research.options_lab import chain_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiquidityStats:
    osi_symbol: str
    asof: str  # ISO date, end of the lookback window
    lookback_days: int
    open_interest: int | None
    bar_volume: int | None
    trade_count: int | None
    source: str = "options_lab_cache"


@dataclass(frozen=True)
class LiquidityTier:
    """Floors matching the live gate in core/option_liquidity.py so research
    results are directly comparable to what the live router would have done."""

    min_open_interest: int = 500
    min_volume: int = 100


DEFAULT_TIER = LiquidityTier()


def contract_liquidity(
    osi_symbol: str, asof: str, *, lookback_days: int = 1, env_file: str | None = ".env"
) -> LiquidityStats | None:
    """Open interest (as of the contract's discovery snapshot) plus summed
    bar volume / trade count over the ``lookback_days`` ending at ``asof``
    (inclusive, ``YYYY-MM-DD``).

    Returns ``None`` only when the contract cannot be resolved at all (bad
    OSI symbol, or the ticker/expiry has zero contract and zero bar rows in
    the cache/API — i.e. the data genuinely does not exist). A contract that
    exists but simply has no OI or no volume on record still returns a
    ``LiquidityStats`` with the corresponding field set to ``None`` so
    callers can distinguish "flat/zero" from "unknown" only where the source
    actually reported it — Alpaca's ``open_interest``/bar ``v``/``n`` fields
    are numeric when present, so a present-but-zero reading is preserved as
    ``0``, and an absent reading is preserved as ``None``.
    """
    try:
        parsed = chain_cache.parse_osi_symbol(osi_symbol)
    except ValueError:
        logger.warning("contract_liquidity: unparseable OSI symbol %r", osi_symbol)
        return None

    contracts = chain_cache.discover_contracts(parsed.root, parsed.expiry, parsed.expiry, env_file=env_file)
    oi: int | None = None
    if not contracts.empty:
        match = contracts.loc[contracts["osi_symbol"] == osi_symbol.upper()]
        if not match.empty:
            raw_oi = match.iloc[0]["open_interest"]
            oi = None if pd.isna(raw_oi) else int(raw_oi)

    end_ts = pd.Timestamp(asof)
    start_ts = end_ts - pd.Timedelta(days=max(0, lookback_days - 1))
    bars = chain_cache.fetch_bars(
        [osi_symbol],
        "1Day",
        start_ts.strftime("%Y-%m-%dT00:00:00Z"),
        end_ts.strftime("%Y-%m-%dT23:59:59Z"),
        env_file=env_file,
    )

    bar_volume: int | None = None
    trade_count: int | None = None
    if not bars.empty:
        sub = bars[bars["osi_symbol"] == osi_symbol.upper()]
        if not sub.empty:
            if "v" in sub.columns and sub["v"].notna().any():
                bar_volume = int(sub["v"].fillna(0).sum())
            if "n" in sub.columns and sub["n"].notna().any():
                trade_count = int(sub["n"].fillna(0).sum())

    if contracts.empty and bars.empty:
        logger.info("contract_liquidity: no contract and no bar data for %s as of %s", osi_symbol, asof)
        return None

    return LiquidityStats(
        osi_symbol=osi_symbol.upper(),
        asof=str(asof),
        lookback_days=lookback_days,
        open_interest=oi,
        bar_volume=bar_volume,
        trade_count=trade_count,
    )


def is_tradable(stats: LiquidityStats | None, tier: LiquidityTier = DEFAULT_TIER) -> bool:
    """``True`` only when both floors are known and met. ``None`` stats or
    ``None`` fields fail closed (not tradable) — never treated as passing."""
    if stats is None:
        return False
    if stats.open_interest is None or stats.bar_volume is None:
        return False
    return stats.open_interest >= tier.min_open_interest and stats.bar_volume >= tier.min_volume


def structure_tradable(legs: Sequence[LiquidityStats | None], tier: LiquidityTier = DEFAULT_TIER) -> bool:
    """Multi-leg gate: the worst leg governs. An empty leg list is not a
    structure, so it is not tradable."""
    if not legs:
        return False
    return all(is_tradable(leg, tier) for leg in legs)


# --------------------------------------------------------------------------
# Spread estimation (uncalibrated placeholder)
# --------------------------------------------------------------------------
#
# TODO(options_lab calibration pass): calibrate the weights below against the
# 16 days of real Schwab bid/ask snapshots in
# Data/dealer_positioning/historical_snapshots/*/dealer_strike_ladder.parquet
# (per plan §6 risk 1). This function currently returns an uncalibrated,
# order-of-magnitude estimate only — do not quote it as a validated spread
# model until that calibration (Gate G1, cross-source check) has run. See
# docs/superpowers/plans/2026-07-25-options-instrument-routing-experiment.md.
_UNCALIBRATED_INTRABAR_WEIGHT = 0.5
_UNCALIBRATED_TRADE_CLUSTER_WEIGHT = 0.5


def estimate_spread_pct(
    osi_symbol: str, asof: str, *, lookback_days: int = 5, env_file: str | None = ".env"
) -> float | None:
    """Rough, **uncalibrated** bid/ask spread estimate as a fraction of price.

    Real historical bid/ask is not available from Alpaca (``/v1beta1/options/
    quotes`` returns 404), so this proxies spread from two observable signals
    over the trailing ``lookback_days`` daily bars ending at ``asof``:

    1. Intrabar dispersion: mean((high - low) / vwap) — a wide high-low range
       relative to vwap suggests a wide effective spread/impact.
    2. Trade-print clustering: dispersion of executed trade prices within the
       window relative to vwap — tightly clustered prints suggest a narrow
       spread, scattered prints suggest a wide one.

    Returns ``None`` when there isn't enough real data (no bars, no vwap) to
    form an estimate — never a fabricated number. The blend weights are
    placeholders (see module TODO above) and must not be treated as
    validated until calibrated against Schwab snapshots.
    """
    end_ts = pd.Timestamp(asof)
    start_ts = end_ts - pd.Timedelta(days=max(0, lookback_days - 1))
    bars = chain_cache.fetch_bars(
        [osi_symbol],
        "1Day",
        start_ts.strftime("%Y-%m-%dT00:00:00Z"),
        end_ts.strftime("%Y-%m-%dT23:59:59Z"),
        env_file=env_file,
    )
    if bars.empty:
        logger.info("estimate_spread_pct: no bar data for %s as of %s", osi_symbol, asof)
        return None
    sub = bars[bars["osi_symbol"] == osi_symbol.upper()]
    sub = sub[sub.get("vw", pd.Series(dtype=float)).notna()] if "vw" in sub.columns else sub.iloc[0:0]
    sub = sub[sub["vw"] > 0] if not sub.empty else sub
    if sub.empty:
        logger.info("estimate_spread_pct: no usable vwap for %s as of %s", osi_symbol, asof)
        return None

    intrabar_dispersion = ((sub["h"] - sub["l"]) / sub["vw"]).mean()

    trades = chain_cache.fetch_trades(
        [osi_symbol],
        start_ts.strftime("%Y-%m-%dT00:00:00Z"),
        end_ts.strftime("%Y-%m-%dT23:59:59Z"),
        env_file=env_file,
    )
    trade_cluster = 0.0
    if not trades.empty and "p" in trades.columns:
        t_sub = trades[trades["osi_symbol"] == osi_symbol.upper()]
        if not t_sub.empty and t_sub["p"].notna().any():
            mean_vwap = sub["vw"].mean()
            if mean_vwap and mean_vwap > 0:
                trade_cluster = float(t_sub["p"].std(ddof=0) / mean_vwap) if len(t_sub) > 1 else 0.0

    estimate = (
        _UNCALIBRATED_INTRABAR_WEIGHT * float(intrabar_dispersion)
        + _UNCALIBRATED_TRADE_CLUSTER_WEIGHT * trade_cluster
    )
    return max(0.0, estimate)
