"""Delisted symbols are skipped before any API call, without touching the universe.

72 universe names (HOLX, QRVO, PSKY, ...) no longer trade but were still fetched
every night: Alpaca bars, every news source, CBOE, the earnings calendar (43
Yahoo 404s on 2026-10-13) and Schwab chains. The list only stops the requests;
eligibility and module scores are deliberately left alone.
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import pytest

import core.shared_universe.delisted as dl

pytestmark = pytest.mark.safe


@pytest.fixture()
def listfile(tmp_path, monkeypatch):
    path = tmp_path / "delisted_symbols.txt"
    path.write_text("# header\n\nHOLX   # acquired\nqrvo\n$PSKY # renamed to SKYD\n", encoding="utf-8")
    monkeypatch.setattr(dl, "DELISTED_SYMBOLS_PATH", path)
    dl._cache.clear()
    yield path
    dl._cache.clear()


def test_parse_ignores_notes_blanks_case_and_dollar_prefix():
    assert dl.parse_delisted_symbols("# c\n\nholx # note\n $QRVO\n") == {"HOLX", "QRVO"}


def test_malformed_line_is_logged_and_skipped_not_raised(caplog):
    with caplog.at_level(logging.ERROR):
        got = dl.parse_delisted_symbols("HOLX\nQRVO PSTG\nWBD\n")
    assert got == {"HOLX", "WBD"}
    assert "one symbol per line" in caplog.text


def test_missing_file_means_nothing_is_delisted(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DELISTED_SYMBOLS_PATH", tmp_path / "absent.txt")
    assert dl.load_delisted_symbols() == frozenset()
    assert not dl.is_delisted("HOLX")


def test_drop_preserves_order_and_logs_once(listfile, caplog):
    with caplog.at_level(logging.INFO):
        kept = dl.drop_delisted(["NVDA", "holx", "AAPL", "PSKY", "MSFT"], context="unit")
    assert kept == ["NVDA", "AAPL", "MSFT"]
    assert caplog.text.count("skipping 2 delisted") == 1


def test_list_edits_are_picked_up_without_a_restart(listfile):
    assert dl.is_delisted("HOLX") and not dl.is_delisted("WBD")
    listfile.write_text("WBD\n", encoding="utf-8")
    later = time.time() + 5
    os.utime(listfile, (later, later))
    assert dl.is_delisted("WBD") and not dl.is_delisted("HOLX")


def test_shipped_list_parses_cleanly_and_spares_core_symbols(caplog):
    with caplog.at_level(logging.ERROR):
        symbols = dl.parse_delisted_symbols(dl.DELISTED_SYMBOLS_PATH.read_text(encoding="utf-8"))
    assert not caplog.text, "every line of the shipped list must parse"
    assert len(symbols) >= 70
    assert not symbols & {"SPY", "QQQ", "IWM", "VIXY", "XLK", "AAPL", "NVDA"}


# --- the providers ---------------------------------------------------------


def test_alpaca_bar_fetch_makes_no_request_for_a_delisted_symbol(listfile, monkeypatch):
    import importlib

    # The package re-exports the function under the module's name; patch the module.
    fi = importlib.import_module("core.API.Alpaca_API.market_data.fetch_intraday")

    def _no_client(*a, **k):
        raise AssertionError("an Alpaca client was created for a delisted symbol")

    monkeypatch.setattr(fi, "StockHistoricalDataClient", _no_client)
    out = fi.fetch_intraday(ticker="HOLX", start="2026-10-01", timeframe="1Day", save_path="")
    assert isinstance(out, pd.DataFrame) and out.empty


def test_bar_cache_is_left_untouched_for_a_delisted_symbol(listfile, monkeypatch, tmp_path):
    from strategies.momentum_expansion.data import bars

    cached = pd.DataFrame({"symbol": ["HOLX"], "timestamp": [pd.Timestamp("2026-04-06", tz="UTC")], "close": [76.0]})
    path = tmp_path / "HOLX.parquet"
    cached.to_parquet(path, index=False)
    monkeypatch.setattr(bars, "_path_for", lambda ticker, kind: path)
    before = path.read_bytes()

    out = bars.fetch_one(ticker="HOLX", kind="1d", end="2026-10-19")

    assert out is not None and len(out) == 1
    assert path.read_bytes() == before


def test_news_cboe_earnings_providers_drop_delisted_tickers(listfile):
    from signals.news import sources

    assert sources._clean_tickers(["aapl", "$HOLX", "QRVO", "MSFT"]) == ["AAPL", "MSFT"]


def test_dealer_symbol_loader_drops_delisted_tickers(listfile):
    from strategies.dealer_positioning.scripts.capture_historical_snapshots import load_symbols

    assert load_symbols(tickers=["NVDA", "HOLX", "AAPL"]) == ["AAPL", "NVDA"]
