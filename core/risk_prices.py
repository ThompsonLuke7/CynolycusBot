"""Timestamp-validated current underlying prices for between-bar stops."""
from __future__ import annotations

import logging
import math
import os

import pandas as pd

logger = logging.getLogger(__name__)


class CurrentUnderlyingPrices:
    """One lazy batch per pass, using the execution client's explicit data feed.

    IEX is the configured default, not a claim of SIP equivalence. An unavailable
    entitlement, stale/future timestamp or unusable quote yields no observation;
    callers retain their explicit unavailable-underlying fallback policy.
    """
    def __init__(self, client, symbols, now, *, feed=None, max_age_seconds=60):
        self.client, self.symbols = client, sorted(set(symbols))
        self.now = pd.Timestamp(now)
        self.feed = feed or os.environ.get("RISK_UNDERLYING_FEED", "iex")
        if self.feed not in {"iex", "sip"}:
            raise ValueError("RISK_UNDERLYING_FEED must be iex or sip; no delayed-feed fallback")
        self.max_age = float(max_age_seconds)
        self.quotes = None

    def get(self, symbol):
        if self.quotes is None:
            try:
                response = self.client.get_stock_quotes(symbols=",".join(self.symbols), feed=self.feed)
                self.quotes = (response or {}).get("quotes", {})
            except Exception as exc:
                logger.warning("Underlying quotes unavailable (feed=%s, %s); premium fallback active",
                               self.feed, type(exc).__name__)
                self.quotes = {}
        q = self.quotes.get(symbol) or {}
        try:
            stamp = pd.Timestamp(q["t"])
            if stamp.tzinfo is None or self.now.tzinfo is None:
                raise ValueError("timezone required")
            age = (self.now - stamp).total_seconds()
            bid, ask = float(q["bp"]), float(q["ap"])
            if not (0 <= age <= self.max_age and 0 < bid <= ask
                    and math.isfinite(bid) and math.isfinite(ask)):
                raise ValueError("stale or invalid quote")
            return (bid + ask) / 2
        except (KeyError, ValueError, TypeError):
            logger.warning("%s current underlying unavailable/invalid (feed=%s); premium fallback active",
                           symbol, self.feed)
            return None
