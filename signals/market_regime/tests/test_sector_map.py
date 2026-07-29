"""Tests for signals.market_regime.sector_map.

D1: no metadata join is available, so unknown tickers must resolve via the
empirical rolling-correlation resolver (behind a flag, D2) or return an
explicit None — never a silent single-ETF fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.market_regime.sector_map import (
    SECTOR_MAP,
    correlate_to_sectors,
    empirical_sector_for,
    sector_etf_for,
)
from signals.market_regime.tests.conftest import make_bars, random_walk_close, session_calendar


def test_curated_map_takes_priority_and_resolver_defaults_off():
    assert sector_etf_for("AAPL") == "XLK"
    assert sector_etf_for("aapl") == "XLK"  # case-insensitive
    # Unknown ticker, resolver not enabled (module default OFF, D2) -> explicit None.
    assert sector_etf_for("ZZZZ_NOT_A_REAL_TICKER") is None


def test_resolver_requires_asof_when_enabled():
    with pytest.raises(ValueError):
        sector_etf_for("ZZZZ_NOT_A_REAL_TICKER", resolver_enabled=True)


def test_correlate_to_sectors_picks_highest_correlation_and_respects_min_obs():
    dates = session_calendar("2021-01-04", 200)
    base = random_walk_close(200, seed=1)
    noise = np.random.default_rng(2).normal(0, 0.001, size=200)
    # Ticker returns track sector "B" almost exactly (small noise); sector "A" is unrelated.
    ticker_close = base * (1 + noise)
    unrelated_close = random_walk_close(200, seed=99)

    ticker_ret = pd.Series(np.log(ticker_close[1:] / ticker_close[:-1]), index=dates[1:])
    sector_b_ret = pd.Series(np.log(base[1:] / base[:-1]), index=dates[1:])
    sector_a_ret = pd.Series(np.log(unrelated_close[1:] / unrelated_close[:-1]), index=dates[1:])

    best_etf, corr, n_obs = correlate_to_sectors(
        ticker_ret, {"SECTOR_A": sector_a_ret, "SECTOR_B": sector_b_ret},
        asof=dates[-1], window_days=120, min_obs=60,
    )
    assert best_etf == "SECTOR_B"
    assert corr > 0.9
    assert n_obs == 120

    # min_obs too high for the available overlap -> no candidate qualifies.
    best_etf_none, corr_none, n_none = correlate_to_sectors(
        ticker_ret, {"SECTOR_A": sector_a_ret, "SECTOR_B": sector_b_ret},
        asof=dates[-1], window_days=120, min_obs=500,
    )
    assert best_etf_none is None
    assert corr_none is None
    assert n_none == 0


def test_correlate_to_sectors_is_point_in_time():
    """Returns dated after `asof` must not influence the correlation — a
    future divergence between the ticker and its true sector must not change
    a past-dated resolution, and the SAME divergence must be visible once
    `asof` moves past it (proving the cutoff is not a no-op)."""
    dates = session_calendar("2021-01-04", 200)
    base = random_walk_close(200, seed=10)
    noise = np.random.default_rng(11).normal(0, 0.0003, size=200)
    ticker_close = base.copy() * (1 + noise)
    # From index 150 on, the ticker's price is redefined by an unrelated,
    # sharp random move (i.e. it stops tracking its true sector) — but this
    # is all in the future relative to the pre-divergence asof used below.
    diverged_tail = ticker_close[149] * np.cumprod(
        1 + np.random.default_rng(99).normal(0, 0.05, size=51)
    )
    ticker_close_diverged = ticker_close.copy()
    ticker_close_diverged[149:] = diverged_tail

    ticker_ret = pd.Series(
        np.log(ticker_close_diverged[1:] / ticker_close_diverged[:-1]), index=dates[1:]
    )
    sector_ret = pd.Series(np.log(base[1:] / base[:-1]), index=dates[1:])
    unrelated_close = random_walk_close(200, seed=77)
    unrelated_ret = pd.Series(np.log(unrelated_close[1:] / unrelated_close[:-1]), index=dates[1:])
    sectors = {"UNRELATED": unrelated_ret, "TRUE_SECTOR": sector_ret}

    # Before the divergence: high correlation with the true sector.
    asof_before = dates[140]
    best_before, corr_before, _ = correlate_to_sectors(
        ticker_ret, sectors, asof=asof_before, window_days=120, min_obs=60,
    )
    assert best_before == "TRUE_SECTOR"
    assert corr_before > 0.95

    # After the divergence has happened: the SAME function, given a later
    # asof, must see a materially lower correlation — proving the cutoff
    # genuinely gates what data is used rather than being a no-op.
    asof_after = dates[-1]
    _, corr_after, _ = correlate_to_sectors(
        ticker_ret, sectors, asof=asof_after, window_days=120, min_obs=60,
    )
    assert corr_after < corr_before - 0.1


def test_empirical_sector_for_caches_monthly_and_is_point_in_time(tmp_path):
    dates = session_calendar("2021-01-04", 200)
    base = random_walk_close(200, seed=20)
    noise = np.random.default_rng(21).normal(0, 0.0005, size=200)
    ticker_close = base * (1 + noise)

    bars = {
        "NEWTICK": make_bars(dates=dates, close=ticker_close),
        "XLK": make_bars(dates=dates, close=base),
    }
    # All other 10 sector ETFs: unrelated random walks so they never win.
    other_etfs = ["XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
    for i, etf in enumerate(other_etfs):
        bars[etf] = make_bars(dates=dates, close=random_walk_close(200, seed=30 + i))

    calls = {"n": 0}

    def counting_loader(ticker):
        calls["n"] += 1
        if ticker not in bars:
            raise FileNotFoundError(ticker)
        return bars[ticker]

    cache_path = tmp_path / "sector_assignments.parquet"
    asof = dates[190]

    result1 = empirical_sector_for("NEWTICK", asof, loader=counting_loader, cache_path=cache_path)
    assert result1 == "XLK"
    calls_after_first = calls["n"]
    assert calls_after_first > 0

    # Second call, same calendar month -> served from the parquet cache, no re-load.
    result2 = empirical_sector_for("NEWTICK", asof, loader=counting_loader, cache_path=cache_path)
    assert result2 == "XLK"
    assert calls["n"] == calls_after_first  # no new loader calls

    cache = pd.read_parquet(cache_path)
    assert len(cache) == 1

    # A different (later) asof in the SAME month reuses the same cached row.
    same_month_later = asof + pd.Timedelta(days=2)
    assert same_month_later.month == asof.month and same_month_later.year == asof.year
    result3 = empirical_sector_for("NEWTICK", same_month_later, loader=counting_loader, cache_path=cache_path)
    assert result3 == "XLK"
    assert calls["n"] == calls_after_first
    cache_after = pd.read_parquet(cache_path)
    assert len(cache_after) == 1


def test_sector_etf_for_uses_empirical_resolver_when_enabled(tmp_path):
    dates = session_calendar("2021-01-04", 200)
    base = random_walk_close(200, seed=40)
    ticker_close = base * (1 + np.random.default_rng(41).normal(0, 0.0005, size=200))
    bars = {"BRANDNEWCO": make_bars(dates=dates, close=ticker_close), "XLV": make_bars(dates=dates, close=base)}
    other_etfs = ["XLK", "XLF", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
    for i, etf in enumerate(other_etfs):
        bars[etf] = make_bars(dates=dates, close=random_walk_close(200, seed=50 + i))

    def loader(ticker):
        if ticker not in bars:
            raise FileNotFoundError(ticker)
        return bars[ticker]

    cache_path = tmp_path / "sector_assignments.parquet"
    # "BRANDNEWCO" is absent from the curated SECTOR_MAP.
    assert "BRANDNEWCO" not in SECTOR_MAP

    result = sector_etf_for(
        "BRANDNEWCO", dates[190], resolver_enabled=True, loader=loader, cache_path=cache_path
    )
    assert result == "XLV"
