"""Unit tests for research/options_lab/liquidity.py.

Mocks chain_cache's public functions directly (no HTTP, no real cache I/O)
and checks the fail-closed / None-vs-zero semantics that matter for
matching core/option_liquidity.py's live gate.
"""
from __future__ import annotations

import pandas as pd

from research.options_lab import chain_cache, liquidity


def _contracts_df(rows):
    cols = ["osi_symbol", "ticker", "expiry", "strike", "right", "open_interest"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _bars_df(rows):
    cols = ["osi_symbol", "t", "o", "h", "l", "c", "v", "n", "vw"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _trades_df(rows):
    cols = ["osi_symbol", "t", "p", "s"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------
# contract_liquidity
# --------------------------------------------------------------------------

def test_contract_liquidity_happy_path(monkeypatch):
    monkeypatch.setattr(
        chain_cache, "discover_contracts",
        lambda ticker, s, e, env_file=None: _contracts_df(
            [{"osi_symbol": "AAPL240216C00185000", "ticker": "AAPL", "expiry": "2024-02-16",
              "strike": 185.0, "right": "C", "open_interest": 1200}]
        ),
    )
    monkeypatch.setattr(
        chain_cache, "fetch_bars",
        lambda symbols, timeframe, start, end, env_file=None: _bars_df(
            [{"osi_symbol": "AAPL240216C00185000", "t": "2024-02-01T00:00:00Z",
              "o": 1, "h": 1.2, "l": 0.9, "c": 1.1, "v": 500, "n": 20, "vw": 1.05},
             {"osi_symbol": "AAPL240216C00185000", "t": "2024-02-02T00:00:00Z",
              "o": 1, "h": 1.3, "l": 0.9, "c": 1.2, "v": 300, "n": 10, "vw": 1.10}]
        ),
    )
    stats = liquidity.contract_liquidity("AAPL240216C00185000", "2024-02-02", lookback_days=2)
    assert stats is not None
    assert stats.open_interest == 1200
    assert stats.bar_volume == 800
    assert stats.trade_count == 30


def test_contract_liquidity_returns_none_when_nothing_exists(monkeypatch):
    monkeypatch.setattr(chain_cache, "discover_contracts", lambda *a, **k: _contracts_df([]))
    monkeypatch.setattr(chain_cache, "fetch_bars", lambda *a, **k: _bars_df([]))
    stats = liquidity.contract_liquidity("ZZZZ240216C00010000", "2024-02-02")
    assert stats is None


def test_contract_liquidity_bad_symbol_returns_none(monkeypatch):
    monkeypatch.setattr(chain_cache, "discover_contracts", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    stats = liquidity.contract_liquidity("not-a-symbol", "2024-02-02")
    assert stats is None


def test_contract_liquidity_missing_oi_stays_none_not_zero(monkeypatch):
    # Contract exists (so we don't hit the "nothing exists" empty path) but
    # open_interest was never reported.
    monkeypatch.setattr(
        chain_cache, "discover_contracts",
        lambda *a, **k: _contracts_df(
            [{"osi_symbol": "AAPL240216C00185000", "ticker": "AAPL", "expiry": "2024-02-16",
              "strike": 185.0, "right": "C", "open_interest": None}]
        ),
    )
    monkeypatch.setattr(chain_cache, "fetch_bars", lambda *a, **k: _bars_df([]))
    stats = liquidity.contract_liquidity("AAPL240216C00185000", "2024-02-02")
    assert stats is not None
    assert stats.open_interest is None
    assert stats.bar_volume is None  # no bars at all -> unknown, not zero


# --------------------------------------------------------------------------
# is_tradable / structure_tradable
# --------------------------------------------------------------------------

def _stats(oi, vol):
    return liquidity.LiquidityStats(
        osi_symbol="X", asof="2024-02-02", lookback_days=1,
        open_interest=oi, bar_volume=vol, trade_count=10,
    )


def test_is_tradable_passes_at_default_floor():
    assert liquidity.is_tradable(_stats(500, 100)) is True


def test_is_tradable_fails_below_floor():
    assert liquidity.is_tradable(_stats(499, 100)) is False
    assert liquidity.is_tradable(_stats(500, 99)) is False


def test_is_tradable_none_stats_is_false():
    assert liquidity.is_tradable(None) is False


def test_is_tradable_none_fields_fail_closed():
    assert liquidity.is_tradable(_stats(None, 100)) is False
    assert liquidity.is_tradable(_stats(500, None)) is False


def test_is_tradable_custom_tier():
    loose = liquidity.LiquidityTier(min_open_interest=10, min_volume=5)
    assert liquidity.is_tradable(_stats(20, 6), loose) is True
    assert liquidity.is_tradable(_stats(20, 6), liquidity.DEFAULT_TIER) is False


def test_structure_tradable_worst_leg_governs():
    good = _stats(1000, 500)
    bad = _stats(10, 5)
    assert liquidity.structure_tradable([good, good]) is True
    assert liquidity.structure_tradable([good, bad]) is False


def test_structure_tradable_empty_legs_is_false():
    assert liquidity.structure_tradable([]) is False


def test_structure_tradable_unknown_leg_is_false():
    assert liquidity.structure_tradable([_stats(1000, 500), None]) is False


# --------------------------------------------------------------------------
# estimate_spread_pct (uncalibrated placeholder)
# --------------------------------------------------------------------------

def test_estimate_spread_pct_none_when_no_bars(monkeypatch):
    monkeypatch.setattr(chain_cache, "fetch_bars", lambda *a, **k: _bars_df([]))
    monkeypatch.setattr(chain_cache, "fetch_trades", lambda *a, **k: _trades_df([]))
    result = liquidity.estimate_spread_pct("AAPL240216C00185000", "2024-02-02")
    assert result is None


def test_estimate_spread_pct_returns_nonnegative_float(monkeypatch):
    monkeypatch.setattr(
        chain_cache, "fetch_bars",
        lambda *a, **k: _bars_df(
            [{"osi_symbol": "AAPL240216C00185000", "t": "2024-02-01T00:00:00Z",
              "o": 1, "h": 1.4, "l": 0.9, "c": 1.1, "v": 500, "n": 20, "vw": 1.05}]
        ),
    )
    monkeypatch.setattr(
        chain_cache, "fetch_trades",
        lambda *a, **k: _trades_df(
            [{"osi_symbol": "AAPL240216C00185000", "t": "2024-02-01T14:00:00Z", "p": 1.02, "s": 5},
             {"osi_symbol": "AAPL240216C00185000", "t": "2024-02-01T14:05:00Z", "p": 1.10, "s": 5}]
        ),
    )
    result = liquidity.estimate_spread_pct("AAPL240216C00185000", "2024-02-02")
    assert result is not None
    assert result >= 0.0


def test_estimate_spread_pct_missing_vwap_returns_none(monkeypatch):
    monkeypatch.setattr(
        chain_cache, "fetch_bars",
        lambda *a, **k: _bars_df(
            [{"osi_symbol": "AAPL240216C00185000", "t": "2024-02-01T00:00:00Z",
              "o": 1, "h": 1.4, "l": 0.9, "c": 1.1, "v": 500, "n": 20, "vw": None}]
        ),
    )
    monkeypatch.setattr(chain_cache, "fetch_trades", lambda *a, **k: _trades_df([]))
    result = liquidity.estimate_spread_pct("AAPL240216C00185000", "2024-02-02")
    assert result is None
