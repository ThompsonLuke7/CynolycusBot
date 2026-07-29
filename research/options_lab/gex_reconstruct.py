"""Reconstruct a historical dealer-positioning (GEX) surface from expired
Alpaca option-contract records + cached daily contract bars.

Why this module exists: `research/options_experiment/04_tail_and_thesis_review.md`
section 4 -- the real Schwab dealer-positioning snapshots
(`Data/dealer_positioning/historical_snapshots/`) only cover 2026-07-02..
2026-07-24 (16 days), which has ZERO overlap with the trade history
(2025-05-20..2026-06-04). The gamma-squeeze mechanism the strategy is built
on has therefore never been tested against a single trade. This module
reconstructs the same call_wall / put_wall / gamma_flip / total_gex levels
over the trade-history window from data Alpaca actually has: `open_interest`
on expired (`status=inactive`) contracts (verified 100% populated), daily
contract bars (volume), and the project's own BSM pricing/IV engine.

THE LOAD-BEARING LIMITATION (do not paper over): Alpaca's `open_interest`
is a SINGLE value dated `open_interest_date` (typically 1 trading day
before expiry), not a daily time series, and it is `null` for contracts
that have not yet expired. A faithful daily OI history is therefore
impossible to reconstruct from this source. This module implements three
explicitly labeled variants (`oi_source`) rather than picking one:

  - "terminal_oi": the near-expiry OI applied as a static estimate for
    every day of the contract's life. Crude but simple; assumes OI barely
    changes day to day (false for names with active new positioning).
  - "volume_accumulated": walks the terminal OI backward using daily
    contract volume -- `oi(t) = max(0, oi_terminal - sum(volume for days
    after t through expiry))`. This is a rough proxy, not a real OI
    process: volume is round-trip (opens AND closes) and this rule
    implicitly treats all post-t volume as net new open interest, which
    over-corrects whenever a name has heavy closing activity.
  - "volume_proxy": no OI at all -- cumulative traded volume up to and
    including the asof date, used directly as the OI-like weight. Immune
    to the terminal-OI-availability gap (works even for still-active
    contracts), but conflates volume with position -- a name that trades
    a lot and closes it all out looks identical to one that holds it.

Validation (not this module -- see `scripts/build_historical_gex.py
--validate` and `research/options_experiment/05_gex_reconstruction.md`)
decides whether any variant is faithful enough to trade on. This module
does not pick a winner.

Dealer convention: gamma exposure is computed via
`strategies.dealer_positioning.levels.build_gamma_ladder` /
`compute_gamma_levels` -- the EXACT functions the live Amethyst/dealer-
positioning module and the real Schwab snapshot capture
(`strategies/dealer_positioning/scripts/capture_historical_snapshots.py`)
use. This module does not reimplement or invent a different sign
convention; it only supplies that function with reconstructed
(strike, right, open_interest, volume, gamma) rows instead of a live
Schwab chain.

This module never talks to Alpaca itself (mirrors the DI discipline in
`research/options_lab/pricing.py`/`surface.py`): every function here takes
plain DataFrames/values. `scripts/build_historical_gex.py` owns the
network/caching side (via `chain_cache.discover_contracts_full` /
`chain_cache.fetch_bars`) and calls into this module per (ticker, date).
"""
from __future__ import annotations

import calendar
import math
from datetime import date, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from research.options_lab import pricing
from strategies.dealer_positioning.levels import build_gamma_ladder, compute_gamma_levels

REQUIRED_SNAPSHOT_COLUMNS = [
    "symbol",
    "date",
    "oi_source",
    "spot",
    "total_gex",
    "net_gex",
    "call_wall",
    "put_wall",
    "gamma_flip",
    "pct_to_call_wall",
    "pct_to_gamma_flip",
    "dealer_bias",
    "total_oi",
    "total_volume",
]

OI_SOURCES = ("terminal_oi", "volume_accumulated", "volume_proxy")


# --------------------------------------------------------------------------
# Expiration-window selection (mirrors UI/dealer_positioning_dashboard.py's
# scope semantics -- reimplemented here as a small, pure, network-free
# helper rather than importing that Tk/live-client-heavy module into
# research code. This is calendar bookkeeping, not the dealer convention;
# the convention itself is reused directly from levels.py, not reinvented.)
# --------------------------------------------------------------------------


def _parse_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


# Public alias -- callers outside this module (e.g. scripts/build_historical_gex.py)
# need the same lenient date/str parsing; exposed rather than duplicated.
parse_date = _parse_date


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + int(months)
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _is_monthly_expiration(value: date) -> bool:
    if value.weekday() != 4:
        return False
    return 15 <= value.day <= 21


def _next_weekly_expiration(dates: list[date]) -> Optional[date]:
    non_monthly_fridays = [d for d in dates if d.weekday() == 4 and not _is_monthly_expiration(d)]
    if non_monthly_fridays:
        return non_monthly_fridays[0]
    fridays = [d for d in dates if d.weekday() == 4]
    return fridays[0] if fridays else None


def _next_monthly_expiration(dates: list[date]) -> Optional[date]:
    monthlies = [d for d in dates if _is_monthly_expiration(d)]
    return monthlies[0] if monthlies else None


def select_expiration_window(
    available_expiries: Iterable, asof, scope: str = "through_month"
) -> list[str]:
    """Return the ISO expiry strings (>= asof) that belong in `scope`,
    mirroring `UI/dealer_positioning_dashboard.py::_select_expirations_for_scope`.

    scope in {"next_expiration", "daily_week", "through_month", "two_months"}.
    """
    asof_d = _parse_date(asof)
    dates = sorted({d for d in (_parse_date(x) for x in available_expiries) if d is not None and d >= asof_d})
    if not dates:
        return []
    if scope == "next_expiration":
        return [dates[0].isoformat()]
    if scope == "two_months":
        cutoff = _add_months(asof_d, 2)
        selected = [d for d in dates if d <= cutoff] or [dates[0]]
        return [d.isoformat() for d in selected]
    if scope == "daily_week":
        target = _next_weekly_expiration(dates) or dates[0]
        return [d.isoformat() for d in dates if d <= target]
    if scope == "through_month":
        target = _next_monthly_expiration(dates) or dates[min(2, len(dates) - 1)]
        return [d.isoformat() for d in dates if d <= target]
    raise ValueError(f"unsupported scope: {scope!r}")


# --------------------------------------------------------------------------
# OI variants
# --------------------------------------------------------------------------


def compute_oi_variants(
    contracts: pd.DataFrame,
    daily_volume_long: pd.DataFrame,
    *,
    asof,
) -> pd.DataFrame:
    """Per-contract OI estimates for all three variants, plus the day's own
    traded volume (used for `total_volume` reporting, identical across
    variants -- it is real data, not an estimate).

    Args:
        contracts: columns [osi_symbol, expiry, terminal_oi (nullable
            float/int), oi_asof (nullable -- open_interest_date, unused
            for the calculation itself but useful for coverage auditing)].
        daily_volume_long: columns [osi_symbol, date, volume] -- one row
            per contract per trading day with bar data. `date` must be
            date-like (not datetime-with-time); duplicate (osi_symbol,
            date) rows are summed.
        asof: the snapshot date.

    Returns `contracts` with added columns:
        day_volume (that day's own traded volume, 0.0 if none),
        volume_to_date (cumulative volume for date <= asof),
        volume_after_date (cumulative volume for asof < date <= expiry),
        oi_terminal_oi, oi_volume_accumulated, oi_volume_proxy.
    """
    required_c = {"osi_symbol", "expiry", "terminal_oi"}
    missing_c = required_c - set(contracts.columns)
    if missing_c:
        raise ValueError(f"contracts missing required columns: {sorted(missing_c)}")
    required_v = {"osi_symbol", "date", "volume"}
    missing_v = required_v - set(daily_volume_long.columns)
    if missing_v:
        raise ValueError(f"daily_volume_long missing required columns: {sorted(missing_v)}")

    asof_d = _parse_date(asof)
    out = contracts.copy().reset_index(drop=True)

    if daily_volume_long.empty:
        out["day_volume"] = 0.0
        out["volume_to_date"] = 0.0
        out["volume_after_date"] = 0.0
    else:
        vol = daily_volume_long.copy()
        vol["date"] = vol["date"].apply(_parse_date)
        vol["volume"] = pd.to_numeric(vol["volume"], errors="coerce").fillna(0.0)
        vol = vol.groupby(["osi_symbol", "date"], as_index=False)["volume"].sum()

        day_volume = vol.loc[vol["date"] == asof_d].set_index("osi_symbol")["volume"]
        to_date = vol.loc[vol["date"] <= asof_d].groupby("osi_symbol")["volume"].sum()

        expiry_by_sym = out.set_index("osi_symbol")["expiry"].apply(_parse_date)
        vol_indexed = vol.set_index("osi_symbol")
        after_vals: dict[str, float] = {}
        for sym, expiry_d in expiry_by_sym.items():
            sub = vol_indexed.loc[vol_indexed.index == sym]
            if sub.empty:
                after_vals[sym] = 0.0
                continue
            mask = (sub["date"] > asof_d) & (sub["date"] <= expiry_d)
            after_vals[sym] = float(sub.loc[mask, "volume"].sum())

        out["day_volume"] = out["osi_symbol"].map(day_volume).fillna(0.0)
        out["volume_to_date"] = out["osi_symbol"].map(to_date).fillna(0.0)
        out["volume_after_date"] = out["osi_symbol"].map(after_vals).fillna(0.0)

    terminal = pd.to_numeric(out["terminal_oi"], errors="coerce")
    out["oi_terminal_oi"] = terminal
    accumulated = terminal - out["volume_after_date"]
    out["oi_volume_accumulated"] = np.where(terminal.notna(), accumulated.clip(lower=0.0), np.nan)
    out["oi_volume_proxy"] = out["volume_to_date"].astype(float)
    return out


_OI_VARIANT_COLUMN = {
    "terminal_oi": "oi_terminal_oi",
    "volume_accumulated": "oi_volume_accumulated",
    "volume_proxy": "oi_volume_proxy",
}


# --------------------------------------------------------------------------
# Gamma + snapshot assembly
# --------------------------------------------------------------------------


def compute_gamma(
    contracts_iv: pd.DataFrame, *, spot: float, r: float, q: float = 0.0, american: bool = False
) -> pd.DataFrame:
    """Add a `gamma` column via `pricing.bsm_greeks`, for every row with a
    non-null `iv` and `T > 0`. Rows with `iv is None` or `T <= 0` get
    `gamma = 0.0` (no convexity contribution), never dropped silently --
    callers can see them via the untouched `iv`/`T` columns.

    `contracts_iv` must have columns [strike, right, T, iv] -- the schema
    `surface.build_iv_surface` returns, or an equivalent synthetic frame.
    """
    required = {"strike", "right", "T", "iv"}
    missing = required - set(contracts_iv.columns)
    if missing:
        raise ValueError(f"contracts_iv missing required columns: {sorted(missing)}")
    out = contracts_iv.copy().reset_index(drop=True)
    gamma = np.zeros(len(out), dtype=float)
    valid = out["iv"].notna() & (out["T"] > 0)
    if valid.any():
        sub = out.loc[valid]
        greeks = pricing.bsm_greeks(
            S=float(spot),
            K=sub["strike"].astype(float).to_numpy(),
            T=sub["T"].astype(float).to_numpy(),
            r=float(r),
            q=float(q),
            sigma=sub["iv"].astype(float).to_numpy(),
            right=sub["right"].tolist(),
        )
        gamma[valid.to_numpy()] = np.asarray(greeks.gamma, dtype=float)
    out["gamma"] = gamma
    return out


def _pct_to(level: Optional[float], spot: float) -> Optional[float]:
    if level is None or spot in (None, 0.0) or (isinstance(level, float) and math.isnan(level)):
        return None
    return (float(level) - float(spot)) / float(spot)


def _dealer_bias(ladder: pd.DataFrame, spot: float) -> Optional[float]:
    """Identical formula to
    `strategies/dealer_positioning/scripts/capture_historical_snapshots.py::_matrix_features`'s
    `dealer_bias`: sum of positive net_gex above spot minus sum of positive
    net_gex below spot."""
    if ladder.empty:
        return None
    above = ladder[ladder["strike"] > spot]
    below = ladder[ladder["strike"] < spot]
    return float(above["net_gex"].clip(lower=0).sum() - below["net_gex"].clip(lower=0).sum())


def assemble_snapshot_row(
    rows: pd.DataFrame,
    *,
    symbol: str,
    date_str: str,
    spot: float,
    oi_source: str,
    total_oi: Optional[float] = None,
    total_volume: Optional[float] = None,
) -> dict:
    """Build one reconstructed-GEX summary row using the project's real
    dealer convention (`strategies.dealer_positioning.levels`), matching
    the schema/semantics of `Data/dealer_positioning/historical_snapshots/
    */dealer_level_summary.parquet` wherever equivalent.

    `rows` must be `OptionContractRow`-compatible: columns
    [strike, option_type ('C'/'P'), open_interest, volume, gamma], plus
    optional [delta, vega, iv, timestamp, expiration, dte].
    """
    if oi_source not in OI_SOURCES:
        raise ValueError(f"oi_source must be one of {OI_SOURCES}, got {oi_source!r}")
    work = rows.copy()
    if "timestamp" not in work.columns:
        work["timestamp"] = date_str
    if "symbol" not in work.columns:
        work["symbol"] = symbol
    ladder, levels = compute_gamma_levels(work, symbol=symbol, spot=float(spot))
    call_wall = levels.call_wall
    put_wall = levels.put_wall
    gamma_flip = levels.gamma_flip
    if total_oi is None:
        total_oi = float(pd.to_numeric(work.get("open_interest"), errors="coerce").fillna(0.0).sum())
    if total_volume is None:
        total_volume = float(pd.to_numeric(work.get("volume"), errors="coerce").fillna(0.0).sum())
    return {
        "symbol": symbol.upper(),
        "date": date_str,
        "oi_source": oi_source,
        "spot": float(spot),
        "total_gex": float(levels.total_gex),
        "net_gex": float(levels.total_gex),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "pct_to_call_wall": _pct_to(call_wall, float(spot)),
        "pct_to_gamma_flip": _pct_to(gamma_flip, float(spot)),
        "dealer_bias": _dealer_bias(ladder, float(spot)),
        "total_oi": total_oi,
        "total_volume": total_volume,
        "n_contracts": int(len(work)),
        "n_strikes": int(ladder["strike"].nunique()) if not ladder.empty else 0,
        "strike_increment": strike_increment(ladder),
    }


def build_rows_for_variant(
    contracts_with_gamma: pd.DataFrame, *, oi_source: str, asof
) -> pd.DataFrame:
    """Filter/rename `contracts_with_gamma` (output of `compute_oi_variants`
    + `compute_gamma`, merged) into `OptionContractRow`-compatible rows for
    one `oi_source` variant, dropping contracts whose OI is unknown for
    that variant (rather than coercing NaN OI to 0 -- an unknown position
    is not a flat one).
    """
    if oi_source not in _OI_VARIANT_COLUMN:
        raise ValueError(f"oi_source must be one of {OI_SOURCES}, got {oi_source!r}")
    oi_col = _OI_VARIANT_COLUMN[oi_source]
    if oi_col not in contracts_with_gamma.columns:
        raise ValueError(f"contracts_with_gamma missing OI column {oi_col!r}; run compute_oi_variants first")
    work = contracts_with_gamma.copy()
    work["open_interest"] = pd.to_numeric(work[oi_col], errors="coerce")
    work = work.dropna(subset=["open_interest"])
    work = work[work["open_interest"] > 0.0]
    work["option_type"] = work["right"].astype(str).str.upper().str[0]
    work["volume"] = pd.to_numeric(work.get("day_volume", 0.0), errors="coerce").fillna(0.0)
    keep = ["strike", "option_type", "open_interest", "volume", "gamma"]
    for extra in ("iv", "delta", "vega", "expiry"):
        if extra in work.columns:
            keep.append(extra)
    out = work[keep].rename(columns={"expiry": "expiration"}) if "expiry" in keep else work[keep]
    return out.reset_index(drop=True)


def strike_increment(ladder_or_strikes) -> Optional[float]:
    """Median gap between adjacent distinct strikes -- used by validation
    to define "within 1 strike" (a fixed +/-2% band is meaningless for a
    $8 stock with $0.50 strikes vs. a $600 stock with $10 strikes)."""
    if isinstance(ladder_or_strikes, pd.DataFrame):
        strikes = ladder_or_strikes["strike"]
    else:
        strikes = pd.Series(list(ladder_or_strikes))
    unique = sorted(set(float(x) for x in pd.to_numeric(strikes, errors="coerce").dropna()))
    if len(unique) < 2:
        return None
    gaps = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    return float(np.median(gaps)) if gaps else None
