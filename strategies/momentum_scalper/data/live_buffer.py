"""Bounded, timestamp-preserving in-memory buffer for live market observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _aware(value: datetime | pd.Timestamp, field: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp.tz_convert("UTC").to_pydatetime()


@dataclass
class LiveMarketBuffer:
    """Store raw observations without overwriting time or feed provenance."""

    max_events_per_kind: int = 200_000
    trades: list[dict[str, Any]] = field(default_factory=list)
    quotes: list[dict[str, Any]] = field(default_factory=list)
    statuses: list[dict[str, Any]] = field(default_factory=list)
    dropped_events: int = 0

    def _append(self, target: list[dict[str, Any]], payload: dict[str, Any]) -> None:
        if len(target) >= self.max_events_per_kind:
            target.pop(0)
            self.dropped_events += 1
        target.append(payload)

    def ingest_trade(self, event: dict[str, Any], *, received_at: datetime) -> None:
        required = {"ticker", "timestamp", "price", "size"}
        missing = sorted(required - set(event))
        if missing:
            raise ValueError(f"trade event missing fields: {missing}")
        payload = dict(event)
        payload["ticker"] = str(payload["ticker"]).upper().strip()
        payload["timestamp"] = _aware(payload["timestamp"], "trade timestamp")
        payload["received_at"] = _aware(received_at, "received_at")
        self._append(self.trades, payload)

    def ingest_quote(self, event: dict[str, Any], *, received_at: datetime) -> None:
        required = {"ticker", "timestamp", "bid", "ask"}
        missing = sorted(required - set(event))
        if missing:
            raise ValueError(f"quote event missing fields: {missing}")
        if float(event["bid"]) <= 0 or float(event["ask"]) < float(event["bid"]):
            raise ValueError("quote must be a positive, non-crossed two-sided market")
        payload = dict(event)
        payload["ticker"] = str(payload["ticker"]).upper().strip()
        payload["timestamp"] = _aware(payload["timestamp"], "quote timestamp")
        payload["received_at"] = _aware(received_at, "received_at")
        self._append(self.quotes, payload)

    def ingest_status(self, event: dict[str, Any], *, received_at: datetime) -> None:
        required = {"ticker", "timestamp", "status"}
        missing = sorted(required - set(event))
        if missing:
            raise ValueError(f"status event missing fields: {missing}")
        payload = dict(event)
        payload["ticker"] = str(payload["ticker"]).upper().strip()
        payload["timestamp"] = _aware(payload["timestamp"], "status timestamp")
        payload["received_at"] = _aware(received_at, "received_at")
        self._append(self.statuses, payload)

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(self.trades), pd.DataFrame(self.quotes), pd.DataFrame(self.statuses)

    def quote_age_seconds(self, *, ticker: str, now: datetime) -> float | None:
        ticker = ticker.upper().strip()
        matching = [row for row in self.quotes if row["ticker"] == ticker]
        if not matching:
            return None
        latest = max(matching, key=lambda row: row["received_at"])
        return max((_aware(now, "now") - latest["received_at"]).total_seconds(), 0.0)


__all__ = ["LiveMarketBuffer"]
