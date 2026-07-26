"""
Live computation of the 192-feature ranker's context features, so the live builder no longer
has a gap vs the offline training matrix.

Two groups:
  * bar-location (~28 feats, intraday, price-derived): computed live by reusing the SAME offline
    functions (signals.location_features) the training export used — see
    models/export_for_colab.py::_build_ticker_location_frame / _build_spy_location_context.
  * daily context (~45 feats: earnings/macro/treasury/news/theme): looked up as-of the prior day
    from daily_context_features.parquet, which matches the training join policy exactly. The parquet
    must be refreshed by the nightly pipeline for these to stay current.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from signals.location_features import add_daily_high_low_gap_features, add_liquidity_zone_features

logger = logging.getLogger(__name__)

# 28 bar-location features (order/identity must match export_for_colab.SWING_LOCATION_FEATURE_COLS)
BAR_LOCATION_COLS = [
    "distance_to_52w_high", "distance_to_52w_low",
    "distance_to_recent_20d_high", "distance_to_recent_20d_low",
    "distance_to_recent_60d_high", "distance_to_recent_60d_low",
    "distance_to_gap_above", "distance_to_gap_below",
    "gap_fill_rate_20d", "gap_fill_rate_60d",
    "breakout_proximity_daily", "support_proximity_daily", "resistance_proximity_daily",
    "distance_to_nearest_support_zone", "distance_to_nearest_resistance_zone",
    "inside_support_zone", "inside_resistance_zone",
    "support_zone_strength", "resistance_zone_strength",
    "failed_breakout_count", "failed_breakdown_count",
    "support_proximity", "resistance_proximity",
    "spy_distance_to_support", "spy_distance_to_resistance",
    "spy_inside_support_zone", "spy_inside_resistance_zone", "spy_regime_score",
]

_ZONE_KW = dict(lookback=78, swing_window=20, zone_width_pct=0.002, volume_window=40)


def _add_location(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    frame, _ = add_daily_high_low_gap_features(frame, prefix=prefix)
    frame, _, _ = add_liquidity_zone_features(
        frame, prefix=prefix,
        atr_col="atr_14" if "atr_14" in frame.columns else None, **_ZONE_KW,
    )
    return frame


def compute_bar_location(price: pd.DataFrame, spy: pd.DataFrame | None) -> dict[str, float]:
    """Return the 28 bar-location features for the LAST bar of `price` (DatetimeIndex OHLCV[+atr_14])."""
    if price is None or price.empty:
        return {}
    frame = _add_location(price.copy())
    # Emit EVERY column, using NaN when the last bar has no value. Dropping the
    # key instead made a legitimately-unavailable value (e.g. gap_fill_rate_60d
    # needs 5 gap days in the trailing 60 sessions) indistinguishable from a
    # builder that never computes the feature at all, so the scanner reported it
    # as "absent from live builder". The model path is unchanged: the scanner
    # reindexes onto its feature list, which NaN-fills either way.
    out: dict[str, float] = {}
    for c in BAR_LOCATION_COLS:
        if c not in frame.columns:
            continue
        value = frame[c].iloc[-1]
        out[c] = float(value) if pd.notna(value) else float("nan")

    if spy is not None and not spy.empty:
        s = _add_location(spy.copy(), prefix="spy_")
        if "market_regime_proxy" in s.columns:
            s["spy_regime_score"] = pd.to_numeric(s["market_regime_proxy"], errors="coerce")
        elif "ret_16" in s.columns:
            r = pd.to_numeric(s["ret_16"], errors="coerce")
            s["spy_regime_score"] = np.where(r > 0.01, 1.0, np.where(r < -0.01, -1.0, 0.0))
        s["spy_distance_to_support"] = s.get("spy_distance_to_nearest_support_zone")
        s["spy_distance_to_resistance"] = s.get("spy_distance_to_nearest_resistance_zone")
        for c in ("spy_distance_to_support", "spy_distance_to_resistance",
                  "spy_inside_support_zone", "spy_inside_resistance_zone", "spy_regime_score"):
            if c in s.columns and pd.notna(s[c].iloc[-1]):
                out[c] = float(s[c].iloc[-1])
    return out


class DailyContextLookup:
    """As-of-prior-day lookup of the daily context features from the nightly parquet."""

    def __init__(self, path: Path | str) -> None:
        self._ok = Path(path).exists()
        if not self._ok:
            logger.warning("DailyContextLookup: %s missing — daily context served as NaN", path)
            self._df = pd.DataFrame()
            self._cols: list[str] = []
            return
        df = pd.read_parquet(path)
        df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        self._cols = [c for c in df.columns if c not in ("ticker", "date")]
        # keep only the latest few rows per ticker for fast as-of lookup
        self._df = df.sort_values(["ticker", "date"])
        self._max_date = df["date"].max()
        logger.info("DailyContextLookup: %d cols, latest date %s", len(self._cols), self._max_date)

    @property
    def columns(self) -> list[str]:
        return self._cols

    def get(self, ticker: str, asof: pd.Timestamp) -> dict[str, float]:
        if not self._ok:
            return {}
        asof = pd.Timestamp(asof).tz_localize(None).normalize() if pd.Timestamp(asof).tzinfo \
            else pd.Timestamp(asof).normalize()
        sub = self._df[(self._df["ticker"] == ticker.upper()) & (self._df["date"] <= asof)]
        if sub.empty:
            return {}
        row = sub.iloc[-1]
        return {c: float(row[c]) for c in self._cols if pd.notna(row[c])}
