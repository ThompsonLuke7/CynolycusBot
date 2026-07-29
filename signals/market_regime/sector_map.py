"""Ticker -> sector-ETF resolution.

Resolution order (plan defect D1 — ``load_candidate_metadata()`` returns
``sector == "Unknown"`` for 100% of the 1125-ticker universe, so a metadata
join is not available):

  1. curated map (hand-maintained, below — copied verbatim from
     ``strategies/momentum_expansion/features/feature_matrix_4h.py``; that
     file's copy is left in place, it is owned by another workstream)
  2. empirical rolling-correlation assignment: 120 trading days of daily log
     returns vs the 11 sector ETFs, recomputed monthly, point-in-time, cached
     to parquet keyed by (ticker, snapshot_month) — behind
     ``config.SECTOR_RESOLVER_ENABLED`` (plan defect D2, default OFF)
  3. ``None`` — explicit unknown, never a silent fallback to one ETF.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategies.momentum_expansion.data.load_bars import load_1d

from .config import (
    SECTOR_ASSIGNMENTS_PATH,
    SECTOR_ETFS_LIST,
    SECTOR_RESOLVER_ENABLED,
    SECTOR_RESOLVER_MIN_OBS,
    SECTOR_RESOLVER_WINDOW_DAYS,
)
from .timeutil import atomic_write_parquet, session_date_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated map — copied verbatim from feature_matrix_4h.py's SECTOR_MAP.
# Do NOT delete the original; another workstream owns that file. Keep these
# two copies in sync manually if the curated list changes.
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    # Tech
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK",
    "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK", "INTC": "XLK", "MU": "XLK",
    "TSM": "XLK", "ASML": "XLK", "ARM": "XLK", "SMCI": "XLK", "PANW": "XLK",
    "CRWD": "XLK", "NET": "XLK", "DDOG": "XLK", "SNOW": "XLK", "PLTR": "XLK",
    "ZS": "XLK", "MSTR": "XLK",
    # Comm / Internet
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "CMCSA": "XLC", "T": "XLC", "VZ": "XLC", "TMUS": "XLC",
    # Cons Disc
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "LOW": "XLY", "NKE": "XLY",
    "MCD": "XLY", "SBUX": "XLY", "F": "XLY", "GM": "XLY", "RIVN": "XLY",
    "LCID": "XLY", "ABNB": "XLY", "UBER": "XLY", "LYFT": "XLY", "DASH": "XLY",
    "BABA": "XLY", "JD": "XLY", "PDD": "XLY", "SHOP": "XLY",
    # Cons Staples
    "WMT": "XLP", "COST": "XLP", "TGT": "XLP", "PG": "XLP", "KO": "XLP",
    "PEP": "XLP", "PM": "XLP", "MO": "XLP",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "V": "XLF", "MA": "XLF", "AXP": "XLF", "PYPL": "XLF",
    "SQ": "XLF", "COIN": "XLF", "SCHW": "XLF",
    # Healthcare
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV",
    "ABBV": "XLV", "TMO": "XLV", "ABT": "XLV",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "OXY": "XLE", "SLB": "XLE",
    "EOG": "XLE", "MPC": "XLE", "PSX": "XLE", "MARA": "XLE", "RIOT": "XLE",
    # Industrials
    "CAT": "XLI", "DE": "XLI", "BA": "XLI", "GE": "XLI", "HON": "XLI",
    "LMT": "XLI", "RTX": "XLI",
}


# ---------------------------------------------------------------------------
# Empirical resolver
# ---------------------------------------------------------------------------

def _daily_log_returns(ticker: str, *, loader=load_1d) -> pd.Series:
    df = loader(ticker)
    dates = session_date_index(df)
    close = pd.Series(df["close"].to_numpy(), index=dates, name=ticker)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    ratio = close / close.shift(1)
    return np.log(ratio.where(ratio > 0))


def _month_start(asof: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(asof)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def correlate_to_sectors(
    ticker_returns: pd.Series,
    sector_returns: dict[str, pd.Series],
    *,
    asof: pd.Timestamp,
    window_days: int = SECTOR_RESOLVER_WINDOW_DAYS,
    min_obs: int = SECTOR_RESOLVER_MIN_OBS,
) -> tuple[str | None, float | None, int]:
    """Point-in-time correlation of ``ticker_returns`` against each sector
    ETF's returns, using only observations dated on/before ``asof`` and the
    trailing ``window_days`` sessions of ``ticker_returns`` within that
    bound. Returns ``(best_sector_etf, correlation, n_obs)``; the sector ETF
    is ``None`` if no candidate clears ``min_obs`` overlapping sessions.

    This is a pure function over supplied series (no disk I/O) so it can be
    unit-tested against a hand-built fixture independent of the on-disk
    monthly cache in ``empirical_sector_for``.
    """
    asof_naive = pd.Timestamp(asof)
    if asof_naive.tzinfo is not None:
        asof_naive = asof_naive.tz_convert("UTC").tz_localize(None)

    t_ret = ticker_returns[ticker_returns.index <= asof_naive].tail(window_days)

    best_etf: str | None = None
    best_corr: float | None = None
    best_n: int = 0
    for etf, s_ret in sector_returns.items():
        s_ret = s_ret[s_ret.index <= asof_naive]
        aligned = pd.concat([t_ret, s_ret], axis=1, join="inner").dropna()
        if len(aligned) < min_obs:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if corr is None or (isinstance(corr, float) and np.isnan(corr)):
            continue
        if best_corr is None or corr > best_corr:
            best_etf, best_corr, best_n = etf, float(corr), len(aligned)
    return best_etf, best_corr, best_n


def _load_assignment_cache(cache_path) -> pd.DataFrame:
    cols = ["ticker", "snapshot_month", "sector_etf", "correlation", "n_obs"]
    if not cache_path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(cache_path)
    df["snapshot_month"] = pd.to_datetime(df["snapshot_month"])
    return df


def empirical_sector_for(
    ticker: str,
    asof: pd.Timestamp | str,
    *,
    loader=load_1d,
    sector_etfs: list[str] = SECTOR_ETFS_LIST,
    window_days: int = SECTOR_RESOLVER_WINDOW_DAYS,
    min_obs: int = SECTOR_RESOLVER_MIN_OBS,
    cache_path=SECTOR_ASSIGNMENTS_PATH,
) -> str | None:
    """Empirical sector assignment for one ticker as of ``asof``, recomputed
    at most once per calendar month (point-in-time snapshot) and cached to
    ``cache_path``. Returns ``None`` if the ticker has no cached bars or
    insufficient overlapping history against any sector ETF.
    """
    ticker = ticker.upper()
    asof_ts = pd.Timestamp(asof)
    snapshot_month = _month_start(asof_ts)

    cache = _load_assignment_cache(cache_path)
    hit = cache[(cache["ticker"] == ticker) & (cache["snapshot_month"] == snapshot_month)]
    if not hit.empty:
        row = hit.iloc[0]
        sector_etf = row["sector_etf"]
        return None if pd.isna(sector_etf) else str(sector_etf)

    try:
        ticker_returns = _daily_log_returns(ticker, loader=loader)
    except FileNotFoundError:
        logger.info("[%s] no cached daily bars — cannot resolve sector empirically", ticker)
        ticker_returns = None

    if ticker_returns is None:
        best_etf, best_corr, n_obs = None, None, 0
    else:
        sector_returns = {}
        for etf in sector_etfs:
            try:
                sector_returns[etf] = _daily_log_returns(etf, loader=loader)
            except FileNotFoundError:
                logger.warning("[%s] no cached daily bars for sector ETF — skipped", etf)
        # Point-in-time cutoff is the snapshot month's first calendar day, not
        # `asof` itself: this makes the monthly-cached assignment identical
        # regardless of which day within the month it is queried on, and
        # guarantees nothing later in the month leaks into the correlation.
        best_etf, best_corr, n_obs = correlate_to_sectors(
            ticker_returns, sector_returns, asof=snapshot_month,
            window_days=window_days, min_obs=min_obs,
        ) if sector_returns else (None, None, 0)

    new_row = pd.DataFrame([{
        "ticker": ticker,
        "snapshot_month": snapshot_month,
        "sector_etf": best_etf,
        "correlation": best_corr,
        "n_obs": n_obs,
    }])
    cache = new_row if cache.empty else pd.concat([cache, new_row], axis=0, ignore_index=True)
    atomic_write_parquet(cache, cache_path, index=False)
    return best_etf


def sector_etf_for(
    ticker: str,
    asof: pd.Timestamp | str | None = None,
    *,
    resolver_enabled: bool | None = None,
    loader=load_1d,
    cache_path=SECTOR_ASSIGNMENTS_PATH,
) -> str | None:
    """Resolve ``ticker`` to a sector ETF: curated map first, then (if
    enabled) the empirical rolling-correlation resolver, else ``None``.

    ``resolver_enabled`` overrides ``config.SECTOR_RESOLVER_ENABLED`` for
    testing / explicit opt-in call sites; the module default is OFF (D2).
    ``loader``/``cache_path`` are forwarded to the empirical resolver so
    callers (and tests) can fully inject data/cache dependencies instead of
    touching the real on-disk bar cache or assignment cache.
    """
    curated = SECTOR_MAP.get(ticker.upper())
    if curated is not None:
        return curated

    enabled = SECTOR_RESOLVER_ENABLED if resolver_enabled is None else resolver_enabled
    if not enabled:
        return None
    if asof is None:
        raise ValueError("asof is required to resolve a sector empirically")
    return empirical_sector_for(ticker, asof, loader=loader, cache_path=cache_path)
