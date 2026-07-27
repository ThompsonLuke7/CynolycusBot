"""Unit tests for research/options_lab/chain_cache.py.

All HTTP is mocked — these tests must not touch the network. They cover:
pagination, cache hit/miss, gap-refetch (resumability), atomic writes, 429
backoff, and the "missing data returns empty, never coerced" contract.
"""
from __future__ import annotations

import json
import urllib.error

import pandas as pd
import pytest

from research.options_lab import chain_cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect the on-disk cache into a tmp dir and stub credentials so no
    test depends on a real .env file or hits the network."""
    monkeypatch.setattr(chain_cache, "CACHE_ROOT", tmp_path / "options_history")
    monkeypatch.setattr(chain_cache, "CONTRACTS_DIR", tmp_path / "options_history" / "contracts")
    monkeypatch.setattr(chain_cache, "BARS_DIR", tmp_path / "options_history" / "bars")
    monkeypatch.setattr(chain_cache, "TRADES_DIR", tmp_path / "options_history" / "trades")
    monkeypatch.setattr(chain_cache, "_credentials", lambda env_file=None: ("key", "secret", "https://data.alpaca.markets"))
    monkeypatch.setattr(chain_cache, "_resolve_trading_base", lambda env_file=None: "https://paper-api.alpaca.markets")
    monkeypatch.setattr(chain_cache.time, "sleep", lambda *_a, **_k: None)
    yield


# --------------------------------------------------------------------------
# OSI symbol parsing
# --------------------------------------------------------------------------

def test_parse_osi_symbol_basic():
    p = chain_cache.parse_osi_symbol("AAPL240216C00185000")
    assert p.root == "AAPL"
    assert p.expiry == "2024-02-16"
    assert p.right == "C"
    assert p.strike == 185.0


def test_parse_osi_symbol_put_and_lowercase():
    p = chain_cache.parse_osi_symbol("aehr260117p00012500")
    assert p.root == "AEHR"
    assert p.expiry == "2026-01-17"
    assert p.right == "P"
    assert p.strike == 12.5


def test_parse_osi_symbol_malformed_raises():
    with pytest.raises(ValueError):
        chain_cache.parse_osi_symbol("not-a-symbol")


# --------------------------------------------------------------------------
# discover_contracts
# --------------------------------------------------------------------------

def _contract_row(symbol, expiry, strike, right, oi):
    return {
        "symbol": symbol,
        "expiration_date": expiry,
        "strike_price": str(strike),
        "type": "call" if right == "C" else "put",
        "open_interest": oi,
    }


def test_discover_contracts_merges_active_and_inactive(monkeypatch):
    calls = []

    def fake_get_json(url, params, *, key, secret):
        calls.append((url, dict(params)))
        status = params["status"]
        if status == "active":
            return {"option_contracts": [_contract_row("AAPL240216C00185000", "2024-02-16", 185, "C", 1200)]}
        return {"option_contracts": [_contract_row("AAPL240119C00180000", "2024-01-19", 180, "C", 300)]}

    monkeypatch.setattr(chain_cache, "_get_json", fake_get_json)
    df = chain_cache.discover_contracts("AAPL", "2024-01-01", "2024-03-01")

    assert set(df["osi_symbol"]) == {"AAPL240216C00185000", "AAPL240119C00180000"}
    assert list(df.columns) == ["osi_symbol", "ticker", "expiry", "strike", "right", "open_interest"]
    # both active and inactive statuses queried
    statuses = {p["status"] for _, p in calls}
    assert statuses == {"active", "inactive"}


def test_discover_contracts_pagination(monkeypatch):
    pages = {
        ("active", None): {
            "option_contracts": [_contract_row("AEHR260117C00010000", "2026-01-17", 10, "C", 50)],
            "next_page_token": "tok1",
        },
        ("active", "tok1"): {
            "option_contracts": [_contract_row("AEHR260117C00012500", "2026-01-17", 12.5, "C", 75)],
            "next_page_token": None,
        },
        ("inactive", None): {"option_contracts": [], "next_page_token": None},
    }

    def fake_get_json(url, params, *, key, secret):
        return pages[(params["status"], params.get("page_token"))]

    monkeypatch.setattr(chain_cache, "_get_json", fake_get_json)
    df = chain_cache.discover_contracts("AEHR", "2026-01-01", "2026-02-01")
    assert len(df) == 2
    assert set(df["strike"]) == {10.0, 12.5}


def test_discover_contracts_cache_hit_makes_no_new_calls(monkeypatch):
    call_count = {"n": 0}

    def fake_get_json(url, params, *, key, secret):
        call_count["n"] += 1
        if params["status"] == "active":
            return {"option_contracts": [_contract_row("AAPL240216C00185000", "2024-02-16", 185, "C", 1200)]}
        return {"option_contracts": []}

    monkeypatch.setattr(chain_cache, "_get_json", fake_get_json)
    df1 = chain_cache.discover_contracts("AAPL", "2024-01-01", "2024-03-01")
    first_calls = call_count["n"]
    assert first_calls == 2  # active + inactive

    df2 = chain_cache.discover_contracts("AAPL", "2024-01-01", "2024-03-01")
    assert call_count["n"] == first_calls  # no new HTTP calls on repeat request
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)


def test_discover_contracts_gap_refetch_only_fetches_missing_window(monkeypatch):
    requested_ranges = []

    def fake_get_json(url, params, *, key, secret):
        requested_ranges.append((params["status"], params["expiration_date_gte"], params["expiration_date_lte"]))
        return {"option_contracts": [], "next_page_token": None}

    monkeypatch.setattr(chain_cache, "_get_json", fake_get_json)

    chain_cache.discover_contracts("AAPL", "2024-01-01", "2024-02-01")
    assert len(requested_ranges) == 2  # active + inactive over Jan window

    requested_ranges.clear()
    # Widen the window: only the new tail (Feb 2 .. Mar 1) should be fetched.
    chain_cache.discover_contracts("AAPL", "2024-01-01", "2024-03-01")
    assert len(requested_ranges) == 2  # active + inactive, but only for the gap
    for _, gte, lte in requested_ranges:
        assert gte >= "2024-02-01"


def test_discover_contracts_no_data_returns_empty_not_none(monkeypatch):
    monkeypatch.setattr(chain_cache, "_get_json", lambda *a, **k: {"option_contracts": [], "next_page_token": None})
    df = chain_cache.discover_contracts("ZZZZ", "2024-01-01", "2024-02-01")
    assert df is not None
    assert df.empty
    assert list(df.columns) == ["osi_symbol", "ticker", "expiry", "strike", "right", "open_interest"]


def test_discover_contracts_missing_open_interest_stays_none_not_zero(monkeypatch):
    def fake_get_json(url, params, *, key, secret):
        if params["status"] == "active":
            row = _contract_row("AAPL240216C00185000", "2024-02-16", 185, "C", None)
            return {"option_contracts": [row], "next_page_token": None}
        return {"option_contracts": [], "next_page_token": None}

    monkeypatch.setattr(chain_cache, "_get_json", fake_get_json)
    df = chain_cache.discover_contracts("AAPL", "2024-01-01", "2024-03-01")
    assert df.iloc[0]["open_interest"] is None


def test_discover_contracts_atomic_write_no_leftover_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(
        chain_cache, "_get_json",
        lambda url, params, **k: (
            {"option_contracts": [_contract_row("AAPL240216C00185000", "2024-02-16", 185, "C", 500)], "next_page_token": None}
            if params["status"] == "active" else {"option_contracts": [], "next_page_token": None}
        ),
    )
    chain_cache.discover_contracts("AAPL", "2024-01-01", "2024-03-01")
    cache_dir = chain_cache.CONTRACTS_DIR
    files = list(cache_dir.iterdir())
    names = {f.name for f in files}
    assert "AAPL.parquet" in names
    assert "AAPL.parquet.meta.json" in names
    assert not any(n.endswith(".tmp") for n in names)
    meta = json.loads((cache_dir / "AAPL.parquet.meta.json").read_text())
    assert "_all_" in meta and len(meta["_all_"]) == 1


# --------------------------------------------------------------------------
# fetch_bars / fetch_trades
# --------------------------------------------------------------------------

def _bar(t, o, h, l, c, v, n, vw):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "n": n, "vw": vw}


def test_fetch_bars_invalid_timeframe_raises():
    with pytest.raises(ValueError):
        chain_cache.fetch_bars(["AAPL240216C00185000"], "5Min", "2024-02-01", "2024-02-16")


def test_fetch_bars_basic_and_grouping_by_ticker_expiry(monkeypatch):
    requests = []

    def fake_fetch(kind, symbols, *, timeframe, start, end, env_file):
        requests.append((kind, tuple(sorted(symbols))))
        return {s: [_bar("2024-02-01T00:00:00Z", 1, 1.2, 0.9, 1.1, 500, 20, 1.05)] for s in symbols}

    monkeypatch.setattr(chain_cache, "_fetch_bars_or_trades_page", fake_fetch)
    df = chain_cache.fetch_bars(
        ["AAPL240216C00185000", "AAPL240216P00185000"], "1Day", "2024-02-01", "2024-02-16"
    )
    assert len(df) == 2
    assert set(df["osi_symbol"]) == {"AAPL240216C00185000", "AAPL240216P00185000"}
    # one grouped multi-symbol batch request for the shared (ticker, expiry)
    assert len(requests) == 1
    assert requests[0][0] == "bars"


def test_fetch_bars_cache_hit_no_refetch(monkeypatch):
    call_count = {"n": 0}

    def fake_fetch(kind, symbols, *, timeframe, start, end, env_file):
        call_count["n"] += 1
        return {s: [_bar("2024-02-01T00:00:00Z", 1, 1.2, 0.9, 1.1, 500, 20, 1.05)] for s in symbols}

    monkeypatch.setattr(chain_cache, "_fetch_bars_or_trades_page", fake_fetch)
    df1 = chain_cache.fetch_bars(["AAPL240216C00185000"], "1Day", "2024-02-01", "2024-02-16")
    assert call_count["n"] == 1
    df2 = chain_cache.fetch_bars(["AAPL240216C00185000"], "1Day", "2024-02-01", "2024-02-16")
    assert call_count["n"] == 1  # fully cached, no new fetch
    assert len(df1) == len(df2) == 1


def test_fetch_bars_gap_refetch_only_new_symbol(monkeypatch):
    seen_symbols = []

    def fake_fetch(kind, symbols, *, timeframe, start, end, env_file):
        seen_symbols.append(tuple(sorted(symbols)))
        return {s: [_bar("2024-02-01T00:00:00Z", 1, 1.2, 0.9, 1.1, 500, 20, 1.05)] for s in symbols}

    monkeypatch.setattr(chain_cache, "_fetch_bars_or_trades_page", fake_fetch)
    chain_cache.fetch_bars(["AAPL240216C00185000"], "1Day", "2024-02-01", "2024-02-16")
    seen_symbols.clear()
    # Second call adds a new symbol in the same (ticker, expiry) file and the
    # same window for the first symbol -> only the new symbol should be fetched.
    chain_cache.fetch_bars(
        ["AAPL240216C00185000", "AAPL240216P00185000"], "1Day", "2024-02-01", "2024-02-16"
    )
    assert seen_symbols == [("AAPL240216P00185000",)]


def test_fetch_bars_no_data_returns_empty_not_none(monkeypatch):
    monkeypatch.setattr(chain_cache, "_fetch_bars_or_trades_page", lambda *a, **k: {})
    df = chain_cache.fetch_bars(["ZZZZ240216C00010000"], "1Day", "2024-02-01", "2024-02-16")
    assert df is not None
    assert df.empty


def test_fetch_trades_basic(monkeypatch):
    def fake_fetch(kind, symbols, *, timeframe, start, end, env_file):
        assert kind == "trades"
        assert timeframe is None
        return {s: [{"t": "2024-02-01T14:30:00Z", "p": 1.05, "s": 10}] for s in symbols}

    monkeypatch.setattr(chain_cache, "_fetch_bars_or_trades_page", fake_fetch)
    df = chain_cache.fetch_trades(["AAPL240216C00185000"], "2024-02-01", "2024-02-16")
    assert len(df) == 1
    assert df.iloc[0]["p"] == 1.05


# --------------------------------------------------------------------------
# 429 backoff (exercises the real _get_json / urllib path)
# --------------------------------------------------------------------------

class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_json_retries_429_then_succeeds(monkeypatch):
    attempts = {"n": 0}
    sleeps = []

    def fake_urlopen(req, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "1"}, None)
        return _FakeHTTPResponse(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(chain_cache.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(chain_cache.time, "sleep", lambda s: sleeps.append(s))

    result = chain_cache._get_json("https://data.alpaca.markets/v1beta1/options/bars", {}, key="k", secret="s")
    assert result == {"ok": True}
    assert attempts["n"] == 3
    assert len(sleeps) == 2


def test_get_json_persistent_429_raises_after_retry_budget(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(chain_cache.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(chain_cache.time, "sleep", lambda s: None)

    with pytest.raises(urllib.error.HTTPError):
        chain_cache._get_json("https://data.alpaca.markets/v1beta1/options/bars", {}, key="k", secret="s")


def test_get_json_non_transient_error_raises_immediately(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(chain_cache.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        chain_cache._get_json("https://data.alpaca.markets/v1beta1/options/quotes", {}, key="k", secret="s")
    assert calls["n"] == 1  # no retry on a non-429 error
