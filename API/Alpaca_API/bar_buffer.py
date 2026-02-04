from __future__ import annotations

from collections import deque
from typing import Iterable

import pandas as pd


class BarRingBuffer:
    """
    In-memory ring buffer for 1-minute bars.

    Each bar is expected to be a mapping with at least:
      - timestamp (datetime or ISO string)
      - open, high, low, close, volume
      - symbol (optional)
    """

    def __init__(self, maxlen: int = 5000) -> None:
        self._buf: deque[dict] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def extend(self, bars: Iterable[dict]) -> None:
        for bar in bars:
            self.append(bar)

    def append(self, bar: dict) -> None:
        self._buf.append(bar)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame indexed by UTC timestamp, sorted ascending.
        """
        if not self._buf:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            ).set_index("timestamp")

        df = pd.DataFrame(self._buf)
        if "timestamp" not in df.columns:
            raise ValueError("Bar buffer is missing required 'timestamp' field.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        df = df.set_index("timestamp")
        return df
