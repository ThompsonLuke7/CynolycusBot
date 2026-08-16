"""Ticker attribution must survive tickers that collide with finance vocabulary.

Every headline below is a REAL row from
signals/news/data/processed/live_catalyst_records.parquet as of 2026-08-03.
`fetch_google_news_rss` queries `"{ticker}" stock` and stamps the result with
that ticker without ever checking the article is about that company.
"""
from __future__ import annotations

import pandas as pd
import pytest

from signals.news.ticker_relevance import classify, filter_frame, is_relevant


# --- the reported case: RSI (Rush Street Interactive) vs the RSI indicator ----

@pytest.mark.parametrize("headline,reason_prefix", [
    ("How to Read and Analyze Stocks: P/E Ratio, RSI, Volume, and More - Binance",
     "ambiguous_ticker_in_indicator_context"),
    ("PayPal Shares Pause as Turnaround Momentum Collides With Overbought RSI - Benzinga",
     "ambiguous_ticker_in_indicator_context"),
    ("These consumer staples stocks show weak RSI ahead of Q2 results (XLP:NYSEARCA) - Seeking Alpha",
     "foreign_symbol_qualified"),
    ("Overbought industrials stocks to watch this Q2 earnings season (XLI:NYSEARCA) - Seeking Alpha",
     "foreign_symbol_qualified"),
    ("Trend Tracker for (RSI.DB.F) (RSI.DB.F:CA) - Stock Traders Daily",
     "foreign_symbol_qualified"),
    ("DBS Stock Gains 2.45% Today, RSI: Full Details - JournalArta",
     "ambiguous_ticker_with_other_symbol"),
])
def test_rsi_indicator_articles_are_rejected(headline, reason_prefix):
    ok, why = classify("RSI", headline)
    assert ok is False
    assert why.startswith(reason_prefix)


@pytest.mark.parametrize("headline", [
    "Rush Street Interactive (RSI) Stock Could Be 3.6% Undervalued After Strong Earnings Beat - simplywall.st",
    "JPMorgan Chase & Co. Issues Positive Forecast for Rush Street Interactive (NYSE:RSI) Stock Price - MarketBeat",
    "Is Rush Street Interactive's Dual Russell 2000 Additions Reframing The Investment Case For RSI? - simplywall.st",
    # No symbol markup at all, but nothing says it is about anything else.
    "RSI Maintained by Wells Fargo -- Price Target Raised to $35.00 - GuruFocus",
    "RSI Maintained by Susquehanna -- Price Target Raised to $36.00 - GuruFocus",
])
def test_genuine_rush_street_articles_are_kept(headline):
    assert is_relevant("RSI", headline) is True


# --- the same defect on other tickers, found while validating the fix --------

@pytest.mark.parametrize("ticker,headline", [
    # "Shares Gap Up/Down" — the word, not Gap Inc.
    ("GAP", "Visteon (NASDAQ:VC) Shares Gap Up on Analyst Upgrade - MarketBeat"),
    ("GAP", "Bloom Energy (NYSE:BE) Shares Gap Down - Should You Sell? - MarketBeat"),
    # "files Form 144" — the word, not FormFactor.
    ("FORM", "Halliburton (NYSE: HAL) files Form 144 noting 24,778 shares and $889K value - Stock Titan"),
    # The job title, not Cooper Companies.
    ("COO", "Meta Platforms (NASDAQ:META) COO Sells $540,576.45 in Stock - MarketBeat"),
    ("COO", "BOX (NYSE:BOX) COO Olivia Nottebohm Sells 5,834 Shares of Stock - MarketBeat"),
    # Canadian ".U" share classes, not Unity.
    ("U", "(QBTC.U) Equity Market Report (QBTC.U:CA) - Stock Traders Daily"),
    # A different company's note, not Wolverine Worldwide.
    ("WWW", "Old Second Bancorp (NASDAQ: OSBC) clears $61.2M stock repurchase plan - Stock Titan"),
])
def test_other_query_artefacts_are_rejected(ticker, headline):
    assert is_relevant(ticker, headline) is False


def test_real_gap_inc_articles_survive():
    assert is_relevant("GAP", "Gap (GAP): Buy, Sell, or Hold Post Q1 Earnings?") is True
    assert is_relevant("GAP", "Gap (GAP) Stock Fair Value Moves Lower As Old Navy Concerns Reset Analyst Views") is True


# --- the filter must not disturb ordinary records ----------------------------

@pytest.mark.parametrize("ticker,headline", [
    ("NVDA", "NVIDIA (NASDAQ:NVDA) Price Target Raised to $250.00 - MarketBeat"),
    ("AAPL", "Apple unveils new MacBook Pro lineup with M5 chip"),
    ("TSLA", "Tesla Q3 deliveries beat expectations at 512,000 vehicles"),
    ("MU", "Micron Technology Announces Pricing of Senior Notes Offering"),
    # Self-qualified alongside another symbol: still ours.
    ("AMD", "AMD (NASDAQ:AMD) and Intel (NASDAQ:INTC) both rally on chip demand"),
])
def test_ordinary_records_are_untouched(ticker, headline):
    assert is_relevant(ticker, headline) is True


def test_an_exchange_name_alone_is_not_a_foreign_symbol():
    """(NASDAQ: ADAM) must read ADAM as the symbol, never NASDAQ as one."""
    ok, _ = classify("ADAM", "ADAMAS TRUST (NASDAQ: ADAM) director awarded 14,238 deferred stock units")
    assert ok is True


def test_empty_inputs_are_kept():
    assert is_relevant("", "some headline") is True
    assert is_relevant("RSI", "") is True
    assert is_relevant("RSI", None) is True


# --- strict mode is opt-in ----------------------------------------------------

def test_strict_mode_rejects_bare_parenthetical_peers_but_default_does_not():
    tfc = "Truist Cuts Boston Scientific (BSX)'s Target as Doctors Change How They Use Its Heart Device"
    assert is_relevant("TFC", tfc) is True                              # default keeps
    assert is_relevant("TFC", tfc, strict_foreign_symbols=True) is False  # strict drops


def test_strict_mode_still_keeps_a_self_qualified_headline():
    h = "Apple (AAPL) gains as Broadcom (AVGO) rallies"
    assert is_relevant("AAPL", h, strict_foreign_symbols=True) is True


# --- frame API ----------------------------------------------------------------

def test_filter_frame_splits_and_annotates():
    df = pd.DataFrame({
        "ticker": ["RSI", "RSI", "NVDA"],
        "headline": [
            "Rush Street Interactive (RSI) Stock Could Be 3.6% Undervalued",
            "PayPal Shares Pause as Turnaround Momentum Collides With Overbought RSI",
            "NVIDIA beats on earnings",
        ],
    })
    kept, dropped = filter_frame(df)
    assert len(kept) == 2 and len(dropped) == 1
    assert dropped.iloc[0]["ticker_relevance_reason"] == "ambiguous_ticker_in_indicator_context"


def test_filter_frame_tolerates_missing_columns_and_empty_frames():
    empty = pd.DataFrame({"ticker": [], "headline": []})
    kept, dropped = filter_frame(empty)
    assert len(kept) == 0 and len(dropped) == 0
    odd = pd.DataFrame({"symbol": ["RSI"]})
    kept, _ = filter_frame(odd)
    assert len(kept) == 1
