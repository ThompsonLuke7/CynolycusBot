"""Timestamp-validated current underlying prices for between-bar stops."""
from __future__ import annotations

import logging
import math
import os

import pandas as pd

logger = logging.getLogger(__name__)

#: Seconds a quote may appear to post-date the fetch clock (exchange vs local clock skew).
CLOCK_SKEW_SECONDS = 5.0
#: A 1-minute bar's close is known at open + 60s.
_BAR_SECONDS = 60.0


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class CurrentUnderlyingPrices:
    """One lazy batch per pass, using the execution client's explicit data feed.

    IEX is the configured default, not a claim of SIP equivalence.

    Age is measured against the time the quotes were FETCHED, not the pass's
    `now`. The risk pass takes `now_et` once and then runs its modules one after
    another, so a liquid name's latest quote routinely post-dates `now_et` by
    1-5 seconds. Measuring against `now_et` gave those quotes a negative age and
    rejected them, and 92% of the 1,123 "premium fallback" events 09-17..09-24
    were exactly that, not missing data
    (research/exit_stop_validation_2026-09-24/07_findings.md §A).

    When the quote is stale, invalid or unavailable, the latest 1-minute bar
    close on the same feed is used if that bar closed within `max_bar_age_seconds`
    (p99 ~3 min on the replayed events, within ~3bp of SIP at the median). Only when both
    fail does `get` return None, and callers keep their explicit fallback.
    """
    def __init__(self, client, symbols, now, *, feed=None, max_age_seconds=60,
                 max_bar_age_seconds=300, clock=None):
        self.client, self.symbols = client, sorted(set(symbols))
        self.now = pd.Timestamp(now)
        self.feed = feed or os.environ.get("RISK_UNDERLYING_FEED", "iex")
        if self.feed not in {"iex", "sip"}:
            raise ValueError("RISK_UNDERLYING_FEED must be iex or sip; no delayed-feed fallback")
        self.max_age = float(max_age_seconds)
        self.max_bar_age = float(max_bar_age_seconds)
        self.clock = clock or _utc_now
        self.quotes = None
        self.quotes_at = None
        self.bars = None
        self.bars_at = None
        #: symbol -> "quote" | "bar" | None, for audit of which source priced the stop.
        self.source: dict[str, str | None] = {}

    def _fetch(self, method):
        """(payload, fetch time). The fetch time is taken AFTER the response."""
        try:
            response = getattr(self.client, method)(symbols=",".join(self.symbols), feed=self.feed)
        except Exception as exc:
            logger.warning("Underlying %s unavailable (feed=%s, %s)", method, self.feed,
                           type(exc).__name__)
            response = None
        return response or {}, max(self.clock(), self.now)

    def _age(self, stamp, fetched_at) -> float:
        stamp = pd.Timestamp(stamp)
        if stamp.tzinfo is None or fetched_at.tzinfo is None:
            raise ValueError("timezone required")
        age = (fetched_at - stamp).total_seconds()
        if age < -CLOCK_SKEW_SECONDS:
            raise ValueError("quote from the future")
        return max(age, 0.0)

    def _from_quote(self, symbol):
        if self.quotes is None:
            payload, self.quotes_at = self._fetch("get_stock_quotes")
            self.quotes = payload.get("quotes", {}) or {}
        q = self.quotes.get(symbol) or {}
        try:
            age = self._age(q["t"], self.quotes_at)
            bid, ask = float(q["bp"]), float(q["ap"])
            if not (age <= self.max_age and 0 < bid <= ask
                    and math.isfinite(bid) and math.isfinite(ask)):
                raise ValueError("stale or invalid quote")
            return (bid + ask) / 2
        except (KeyError, ValueError, TypeError):
            return None

    def _from_bar(self, symbol):
        if self.bars is None:
            if not hasattr(self.client, "get_latest_stock_bars"):
                self.bars, self.bars_at = {}, self.now
            else:
                payload, self.bars_at = self._fetch("get_latest_stock_bars")
                self.bars = payload.get("bars", {}) or {}
        b = self.bars.get(symbol) or {}
        try:
            # `t` is the bar OPEN; its close price is at most 60s after that (an
            # in-progress bar's close is the latest trade), so age runs from the
            # close time and is 0 for a bar still forming.
            opened_age = self._age(b["t"], self.bars_at)
            age = max(0.0, opened_age - _BAR_SECONDS)
            close = float(b["c"])
            if not (age <= self.max_bar_age and close > 0 and math.isfinite(close)):
                raise ValueError("stale or invalid bar")
            return close
        except (KeyError, ValueError, TypeError):
            return None

    def get(self, symbol):
        px = self._from_quote(symbol)
        if px is not None:
            self.source[symbol] = "quote"
            return px
        px = self._from_bar(symbol)
        if px is not None:
            self.source[symbol] = "bar"
            logger.info("%s underlying quote stale/invalid (feed=%s); using latest 1-min bar close",
                        symbol, self.feed)
            return px
        self.source[symbol] = None
        logger.warning("%s current underlying unavailable/invalid (feed=%s, quote and bar); "
                       "premium fallback active", symbol, self.feed)
        return None
