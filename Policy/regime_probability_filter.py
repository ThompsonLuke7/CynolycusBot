from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Policy.regime_filter import StickyRegimeConfig, add_sticky_trend_regime


@dataclass(frozen=True)
class RegimeProbabilityThresholdConfig:
    bullish_long_quantile: float = 0.85
    bullish_short_quantile: float = 0.98
    bearish_long_quantile: float = 0.98
    bearish_short_quantile: float = 0.85
    neutral_long_quantile: float = 0.95
    neutral_short_quantile: float = 0.95
    min_threshold: float = 0.05
    max_threshold: float = 0.95

    def quantile_for(self, *, regime: str, side: str) -> float:
        regime_key = str(regime or "neutral").strip().lower()
        side_key = str(side or "").strip().lower()
        attr = f"{regime_key}_{side_key}_quantile"
        value = getattr(self, attr, None)
        if value is None:
            value = getattr(self, f"neutral_{side_key}_quantile", 0.90)
        return float(np.clip(float(value), 0.0, 1.0))


class RegimeProbabilityCalibrator:
    def __init__(
        self,
        *,
        distributions: dict[tuple[str, str], np.ndarray],
        threshold_config: RegimeProbabilityThresholdConfig | None = None,
    ) -> None:
        self._distributions = {
            (str(regime), str(side)): np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
            for (regime, side), values in distributions.items()
        }
        self._threshold_config = threshold_config or RegimeProbabilityThresholdConfig()

    @classmethod
    def from_signal_frame(
        cls,
        path: str | Path,
        *,
        threshold_config: RegimeProbabilityThresholdConfig | None = None,
        regime_config: StickyRegimeConfig | None = None,
    ) -> "RegimeProbabilityCalibrator":
        frame = pd.read_parquet(path)
        if not isinstance(frame.index, pd.DatetimeIndex):
            if "timestamp" not in frame.columns:
                raise ValueError("Regime probability calibration frame needs DatetimeIndex or timestamp column.")
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["timestamp"]).set_index("timestamp")
        idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.loc[pd.notna(idx)].copy()
        frame.index = pd.DatetimeIndex(idx[pd.notna(idx)]).tz_convert("America/New_York")
        for col in ("open", "high", "low", "close", "ema_fast", "ema_slow"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = add_sticky_trend_regime(frame, config=regime_config or StickyRegimeConfig())
        frame["p_long"] = _combine_prob(frame, "long")
        frame["p_short"] = _combine_prob(frame, "short")
        distributions: dict[tuple[str, str], np.ndarray] = {}
        for regime, group in frame.groupby("trend_regime"):
            for side in ("long", "short"):
                values = pd.to_numeric(group[f"p_{side}"], errors="coerce").dropna().to_numpy(dtype=float)
                distributions[(str(regime), side)] = values
        return cls(distributions=distributions, threshold_config=threshold_config)

    def percentile(self, *, regime: str, side: str, probability: Any) -> float:
        value = _as_float(probability)
        values = self._values(regime=regime, side=side)
        if not np.isfinite(value) or values.size == 0:
            return float("nan")
        return float(np.searchsorted(values, value, side="right") / values.size)

    def threshold(self, *, regime: str, side: str) -> float:
        q = self._threshold_config.quantile_for(regime=regime, side=side)
        values = self._values(regime=regime, side=side)
        if values.size == 0:
            return float("nan")
        raw = float(np.quantile(values, q))
        return float(np.clip(raw, self._threshold_config.min_threshold, self._threshold_config.max_threshold))

    def annotate_row(self, row: pd.Series) -> dict[str, float | str | None]:
        regime = str(row.get("trend_regime", "neutral") or "neutral")
        p_long = _as_float(row.get("p_enter_long", row.get("p_swing_setup_long", np.nan)))
        p_short = _as_float(row.get("p_enter_short", row.get("p_swing_setup_short", np.nan)))
        return {
            "trend_regime": regime,
            "p_enter_long_regime_pctile": self.percentile(regime=regime, side="long", probability=p_long),
            "p_enter_short_regime_pctile": self.percentile(regime=regime, side="short", probability=p_short),
            "regime_thr_enter_long": self.threshold(regime=regime, side="long"),
            "regime_thr_enter_short": self.threshold(regime=regime, side="short"),
        }

    def _values(self, *, regime: str, side: str) -> np.ndarray:
        regime_key = str(regime or "neutral").strip().lower()
        side_key = str(side or "").strip().lower()
        values = self._distributions.get((regime_key, side_key))
        if values is None or values.size == 0:
            values = self._distributions.get(("neutral", side_key))
        if values is None:
            return np.array([], dtype=float)
        return values


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _combine_prob(frame: pd.DataFrame, side: str) -> pd.Series:
    candidates = [
        f"p_{side}_full",
        f"p_{side}_test",
        f"p_{side}_oof_train",
        f"p_swing_setup_{side}",
        f"p_enter_{side}",
    ]
    out: pd.Series | None = None
    for col in candidates:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        out = values if out is None else out.combine_first(values)
    if out is None:
        out = pd.Series(index=frame.index, dtype=float)
    return out
