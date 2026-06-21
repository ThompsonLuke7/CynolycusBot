"""Live catalyst signal for the swing scanner.

Reads the intraday catalyst ledger produced by ``scripts/intraday_catalyst_poll.py``
(``live_catalyst_records.parquet``) and exposes a recency-weighted per-ticker
catalyst score so the scanner can react to fresh news in real time.

``catalyst_score`` is the binary catalyst model's P(bullish ~10% expansion), so it
is long-directional: a high score supports longs and argues against shorts. The
reader is defensive — a missing/empty/unreadable ledger yields a NEUTRAL hit and
the scanner behaves exactly as it did before news wiring.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LIVE_LEDGER_PATH = Path("signals/news/data/processed/live_catalyst_records.parquet")
_COLS = ["ticker", "timestamp", "catalyst_score", "headline", "catalyst_family"]


@dataclass(frozen=True)
class CatalystHit:
    score: float                       # recency-weighted catalyst strength in [0,1]
    count: int                         # records inside the lookback window
    latest_ts: pd.Timestamp | None
    top_headline: str | None
    top_family: str | None


NEUTRAL = CatalystHit(score=float("nan"), count=0, latest_ts=None, top_headline=None, top_family=None)


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def catalyst_ev_multiplier(hit: CatalystHit | None, direction: int, *, alpha: float, baseline: float = 0.5) -> float:
    """Direction-aware multiplicative tilt for a signal's EV score.

    Bullish catalysts (score > baseline) boost longs and dampen shorts. Returns
    1.0 (no effect) when there is no fresh catalyst. Clamped to stay positive so
    a positive-EV signal can never be flipped negative by the tilt alone.
    """
    if hit is None or hit.count == 0 or not np.isfinite(hit.score):
        return 1.0
    return float(max(0.05, 1.0 + alpha * (hit.score - baseline) * float(direction)))


class LiveCatalystSignal:
    """Cached reader over the live catalyst ledger.

    Reloads at most once per ``refresh_seconds`` and only when the file mtime
    changes, so per-bar scans across the universe stay cheap.
    """

    def __init__(
        self,
        ledger_path: Path | str = LIVE_LEDGER_PATH,
        *,
        lookback_minutes: int = 180,
        halflife_minutes: float = 45.0,
        refresh_seconds: float = 60.0,
    ) -> None:
        self._path = Path(ledger_path)
        self._lookback = pd.Timedelta(minutes=lookback_minutes)
        self._halflife = float(halflife_minutes)
        self._refresh = float(refresh_seconds)
        self._df: pd.DataFrame | None = None
        self._loaded_mtime: float | None = None
        self._last_check = 0.0

    def _maybe_reload(self) -> None:
        now = time.monotonic()
        if self._df is not None and (now - self._last_check) < self._refresh:
            return
        self._last_check = now
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            self._df = None
            self._loaded_mtime = None
            return
        if self._loaded_mtime == mtime and self._df is not None:
            return
        try:
            df = pd.read_parquet(self._path, columns=_COLS)
        except Exception as exc:  # noqa: BLE001 - never let a bad ledger break scanning
            logger.warning("catalyst ledger read failed (%s): %s", self._path, exc)
            self._df = None
            self._loaded_mtime = None
            return
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["catalyst_score"] = pd.to_numeric(df["catalyst_score"], errors="coerce")
        self._df = df.dropna(subset=["ticker", "timestamp", "catalyst_score"])
        self._loaded_mtime = mtime

    def for_ticker(self, ticker: str, *, now=None) -> CatalystHit:
        self._maybe_reload()
        if self._df is None or self._df.empty:
            return NEUTRAL
        now_ts = _as_utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
        sub = self._df[self._df["ticker"] == str(ticker).upper()]
        if sub.empty:
            return NEUTRAL
        sub = sub[sub["timestamp"] >= (now_ts - self._lookback)]
        if sub.empty:
            return NEUTRAL
        age_min = (now_ts - sub["timestamp"]).dt.total_seconds().to_numpy() / 60.0
        w = np.power(0.5, np.clip(age_min, 0.0, None) / self._halflife)
        scores = sub["catalyst_score"].to_numpy(float)
        score = float(np.average(scores, weights=w)) if w.sum() > 0 else float(scores.mean())
        top = sub.loc[sub["catalyst_score"].idxmax()]
        return CatalystHit(
            score=score,
            count=int(len(sub)),
            latest_ts=sub["timestamp"].max(),
            top_headline=(str(top["headline"]) if "headline" in sub.columns and pd.notna(top["headline"]) else None),
            top_family=(str(top["catalyst_family"]) if "catalyst_family" in sub.columns and pd.notna(top["catalyst_family"]) else None),
        )
