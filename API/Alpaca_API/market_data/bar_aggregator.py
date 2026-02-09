from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class AggregatedBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol,
        }


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def floor_to_interval(ts: datetime, minutes: int) -> datetime:
    ts = _to_utc(ts).replace(second=0, microsecond=0)
    minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=minute)


def ceil_to_interval(ts: datetime, minutes: int) -> datetime:
    ts = _to_utc(ts).replace(second=0, microsecond=0)
    minute = ((ts.minute + minutes - 1) // minutes) * minutes
    if minute == 60:
        ts = ts + timedelta(hours=1)
        minute = 0
    return ts.replace(minute=minute)


class OhlcvAggregator:
    """
    Aggregate 1-minute bars into larger OHLCV buckets (default 15m).

    By default, buckets are labeled by their *start* time (left-closed, left-labeled).
    """

    def __init__(self, interval_minutes: int = 15, *, label: str = "left") -> None:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive.")
        label = label.lower().strip()
        if label not in {"left", "right"}:
            raise ValueError("label must be 'left' or 'right'.")
        self.interval_minutes = interval_minutes
        self._label = label
        self._bucket_start: Optional[datetime] = None
        self._agg: Optional[AggregatedBar] = None

    @property
    def current_bucket_start(self) -> Optional[datetime]:
        return self._bucket_start

    def reset(self) -> None:
        self._bucket_start = None
        self._agg = None

    def update(self, bar: dict) -> tuple[Optional[dict], Optional[dict]]:
        """
        Update aggregation with a new 1m bar.

        Returns:
          (closed_bar, current_bar)
          - closed_bar is the completed interval bar (if a bucket rolled).
          - current_bar is the in-progress aggregate after applying this bar.
        """
        ts = bar.get("timestamp")
        if ts is None:
            raise ValueError("bar is missing 'timestamp'")
        if not isinstance(ts, datetime):
            raise TypeError("bar['timestamp'] must be a datetime")

        symbol = bar.get("symbol")
        bucket = floor_to_interval(ts, self.interval_minutes)

        if self._bucket_start is None:
            self._bucket_start = bucket
            self._agg = AggregatedBar(
                timestamp=bucket,
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["volume"]),
                symbol=symbol,
            )
            return None, self._agg.to_dict()

        if bucket < self._bucket_start:
            # Out-of-order bar; ignore to keep aggregation monotonic.
            return None, self._agg.to_dict() if self._agg else None

        if bucket != self._bucket_start:
            closed = self._agg.to_dict() if self._agg else None
            if closed and self._label == "right":
                closed["timestamp"] = closed["timestamp"] + timedelta(
                    minutes=self.interval_minutes
                )
            self._bucket_start = bucket
            self._agg = AggregatedBar(
                timestamp=bucket,
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["volume"]),
                symbol=symbol,
            )
            return closed, self._agg.to_dict()

        if self._agg is None:
            self._agg = AggregatedBar(
                timestamp=bucket,
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["volume"]),
                symbol=symbol,
            )
            return None, self._agg.to_dict()

        self._agg.high = max(self._agg.high, float(bar["high"]))
        self._agg.low = min(self._agg.low, float(bar["low"]))
        self._agg.close = float(bar["close"])
        self._agg.volume += float(bar["volume"])
        current = self._agg.to_dict()
        if self._label == "right":
            current["timestamp"] = current["timestamp"] + timedelta(
                minutes=self.interval_minutes
            )
        return None, current
