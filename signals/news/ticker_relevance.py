"""Reject news records whose ticker attribution is an artefact of the query.

`fetch_google_news_rss` queries Google for `"{ticker}" stock` and stamps every
result with that ticker. Nothing checks that the article is actually ABOUT that
company, so any ticker colliding with finance vocabulary collects systematic
false positives. RSI (Rush Street Interactive, NYSE:RSI) is the worst offender
in the live ledger — 9 of its 15 records on 2026-08-03 were about the Relative
Strength Index, a different listed instrument, or another company entirely:

    "How to Read and Analyze Stocks: P/E Ratio, RSI, Volume, and More"
    "These consumer staples stocks show weak RSI ahead of Q2 results (XLP:NYSEARCA)"
    "PayPal Shares Pause as Turnaround Momentum Collides With Overbought RSI"
    "Trend Tracker for (RSI.DB.F) (RSI.DB.F:CA)"        <- a Canadian debenture

Two structural rules, both deliberately conservative — no company-name map is
available in this repo, so the filter only rejects on POSITIVE evidence that the
headline is about something else, never on absence of evidence:

  A. Foreign symbol qualification (all tickers). The headline names one or more
     exchange-qualified symbols and none of them is ours.
  B. Ambiguous ticker in indicator context (curated list only). The ticker is a
     known finance term, the headline carries technical-indicator vocabulary or
     names a different ticker-shaped token, and nothing self-qualifies it as the
     company.

Anything not matching a rule is kept, so a legitimate headline with no explicit
symbol markup ("RSI Maintained by Wells Fargo -- Price Target Raised to $35")
survives.
"""
from __future__ import annotations

import re

# Tickers that are also common finance/technical vocabulary. Only these are
# eligible for rule B; every other ticker is subject to rule A alone.
AMBIGUOUS_TICKERS: frozenset[str] = frozenset({
    # technical indicators / market jargon
    "RSI", "ATR", "MACD", "VWAP", "EPS", "PE", "PEG", "ROE", "ROI", "ROIC",
    "EV", "IV", "OI", "SMA", "EMA", "ADX", "CCI", "MFI", "OBV", "TTM",
    # macro / general finance
    "GDP", "CPI", "PPI", "FED", "IPO", "ETF", "REIT", "APR", "AUM", "NAV",
    # everyday English words that are also listed symbols
    "ALL", "ARE", "CAN", "FOR", "GO", "HAS", "IT", "NOW", "ON", "SO", "BE",
    "AI", "OR", "AN", "BY", "UP", "DD", "TRUE", "OPEN", "KEY", "LOVE", "CARS",
})

# Technical-analysis vocabulary. Presence alongside an ambiguous ticker is
# evidence the ticker is being used as an indicator, not as a company.
_INDICATOR_CONTEXT = re.compile(
    r"\b("
    r"overbought|oversold|relative\s+strength|moving\s+average|bollinger|"
    r"macd|stochastic|fibonacci|candlestick|chart\s+pattern|support\s+and\s+resistance|"
    r"p/?e\s+ratio|price[-\s]to[-\s]earnings|technical\s+(?:analysis|indicator|setup)|"
    r"how\s+to\s+(?:read|analyze|trade)|indicator[s]?\b|momentum\s+oscillator"
    r")\b",
    re.IGNORECASE,
)

# (NYSE:ABC) / (ABC:NYSEARCA) / (NASDAQ: ABC) / (RSI.DB.F:CA) — an explicitly
# qualified instrument reference. Captures both orderings.
_QUALIFIED_SYMBOL = re.compile(
    r"\(\s*([A-Z][A-Z0-9.\-]{0,14})\s*:\s*([A-Z][A-Z0-9.\-]{0,14})\s*\)"
)
# A bare parenthetical symbol: "Rush Street Interactive (RSI) Stock ..."
_BARE_PAREN_SYMBOL = re.compile(r"\(\s*([A-Z][A-Z0-9.\-]{0,14})\s*\)")
# Ticker-shaped standalone tokens, for rule B's "names another company" check.
_TICKER_TOKEN = re.compile(r"\b([A-Z]{2,5})\b")

# Uppercase tokens that are ticker-shaped but are never a company reference in a
# headline, so they must not trip rule B's "another ticker is named" check.
_TOKEN_STOPWORDS: frozenset[str] = frozenset({
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BUT", "BY", "FOR", "FROM", "HAS",
    "HOW", "IN", "IS", "IT", "ITS", "MORE", "NEW", "NOT", "NOW", "OF", "ON", "OR",
    "OUR", "OUT", "SO", "THE", "THIS", "TO", "UP", "VS", "WE", "WHAT", "WHY",
    "WILL", "WITH", "YOU", "Q1", "Q2", "Q3", "Q4", "CEO", "CFO", "COO", "USA",
    "US", "UK", "EU", "AI", "IPO", "ETF", "NYSE", "NASDAQ", "NYSEARCA", "AMEX",
    "OTC", "TSX", "LSE", "SEC", "FDA", "FED", "GDP", "CPI", "EPS", "PE", "PT",
    "BUY", "SELL", "HOLD", "STOCK", "SHARES", "INC", "CORP", "LTD", "PLC", "CO",
    "DAILY", "FULL", "TODAY", "WEEK", "YEAR", "HERE", "WHATS", "MAY", "CAN",
})


def _symbols_in(headline: str) -> tuple[set[str], set[str]]:
    """(qualified_symbols, bare_parenthetical_symbols) found in the headline."""
    qualified: set[str] = set()
    for a, b in _QUALIFIED_SYMBOL.findall(headline):
        # Either ordering: (NYSE:RSI) or (XLP:NYSEARCA). Keep both sides; the
        # caller only asks "is our ticker among these", so extra entries are safe.
        qualified.update({a.upper(), b.upper()})
    bare = {m.upper() for m in _BARE_PAREN_SYMBOL.findall(headline)}
    return qualified, bare


_EXCHANGE_NAMES = frozenset({"NYSE", "NASDAQ", "NYSEARCA", "AMEX", "OTC", "OTCMKTS",
                             "TSX", "TSXV", "CSE", "LSE", "ASX", "CA", "US"})


def classify(ticker: str, headline: str, *, strict_foreign_symbols: bool = False) -> tuple[bool, str]:
    """(is_relevant, reason). reason is "ok" when the record is kept.

    `strict_foreign_symbols` also treats a BARE parenthetical symbol — "Truist
    Cuts Boston Scientific (BSX)'s Target", tagged TFC — as foreign evidence.
    Measured on the 18,605-record live ledger (2026-08-03): the default drops
    296 rows (1.59%) and leaves mega-caps at 0.0-0.3%; strict drops 877 (4.71%)
    and takes GOOG to 7.7%, AMZN 3.9%, AAPL 3.0%. Most of that extra is genuine
    misattribution, but some is a peer-mention article that does discuss our
    name. That is a signal-design judgement about what should count as a
    catalyst, not a bug, so it is OFF until someone evaluates it downstream.
    """
    tk = str(ticker or "").strip().upper()
    text = str(headline or "").strip()
    if not tk or not text:
        return True, "ok"

    qualified, bare = _symbols_in(text)
    self_qualified = tk in qualified or tk in bare

    # --- Rule A: the headline explicitly qualifies OTHER instruments, not ours.
    # Exchange names are not instruments, so ignore them when deciding whether
    # any real foreign symbol was named.
    candidates = (qualified | bare) if strict_foreign_symbols else qualified
    foreign_qualified = {s for s in candidates if s not in _EXCHANGE_NAMES and s != tk}
    if foreign_qualified and not self_qualified:
        return False, f"foreign_symbol_qualified:{','.join(sorted(foreign_qualified))}"

    if tk not in AMBIGUOUS_TICKERS or self_qualified:
        return True, "ok"

    # --- Rule B (ambiguous tickers only, and only when nothing self-qualifies).
    if _INDICATOR_CONTEXT.search(text):
        return False, "ambiguous_ticker_in_indicator_context"

    other_tokens = {
        t for t in _TICKER_TOKEN.findall(text)
        if t != tk and t not in _TOKEN_STOPWORDS and t not in _EXCHANGE_NAMES
    }
    if other_tokens and tk in _TICKER_TOKEN.findall(text):
        # Our ambiguous ticker appears bare alongside another ticker-shaped
        # name, with nothing marking ours as the subject.
        return False, f"ambiguous_ticker_with_other_symbol:{','.join(sorted(other_tokens))}"

    return True, "ok"


def is_relevant(ticker: str, headline: str, **kw) -> bool:
    return classify(ticker, headline, **kw)[0]


def filter_frame(df, *, ticker_col: str = "ticker", headline_col: str = "headline",
                 reason_col: str | None = "ticker_relevance_reason",
                 strict_foreign_symbols: bool = False):
    """Drop query-artefact rows from a records frame. Returns (kept, dropped)."""
    if df is None or len(df) == 0 or ticker_col not in df or headline_col not in df:
        return df, df.iloc[0:0] if df is not None else df
    verdicts = [classify(t, h, strict_foreign_symbols=strict_foreign_symbols)
                for t, h in zip(df[ticker_col], df[headline_col])]
    keep = [v[0] for v in verdicts]
    dropped = df[[not k for k in keep]].copy()
    if reason_col and len(dropped):
        dropped[reason_col] = [v[1] for v, k in zip(verdicts, keep) if not k]
    return df[keep].copy(), dropped
