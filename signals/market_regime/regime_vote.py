"""Rule-based market regime: a seven-signal bearish vote over one daily-regime row.

WHY. The nervous-system ``MarketState`` has carried ``MarketRegime.UNKNOWN`` on
every published row, so every governed decision got the flat UNKNOWN size
multiplier no matter what the inputs said (it read UNKNOWN through the whole
Oct-Nov 2026 selloff while its own metrics were flashing red).  This module is
the classifier that fills that field.

It is deliberately a handwritten rule, not a model, and it is the baseline any
learned regime model must beat.  It was tested before adoption in
research/regime_switch_2026-11-20/ (05_intraday_structure_by_regime.py,
07_vote_vs_sma50.py, README section 7): against QQQ < SMA50 alone it had about
half the false flags (1.1 vs 1.9 per year, 2020-08..2026-11) and flagged the
real selloffs on the same day.  It is a state description, NOT a forecast of
direction: shorting on it alone was net negative in the same study.

Every vote uses values known at the close of the row's session; the row's
``available_at`` is when the vote becomes usable (the next session's decisions).

Votes (1 = bearish):
  qqq_fast_down    QQQ close < EMA20 and EMA20 below its value 5 sessions ago
  qqq_below_sma50  QQQ close < SMA50
  spy_trend        spy_trend_state == -1 (SPY EMA20 < EMA50)
  breadth          breadth_z < 0
  appetite         risk_appetite_z < 0
  credit           credit_risk_z < 0 (HYG/IEI; duration-controlled)
  vol              spy_rv20_z > 1

Mapping (thresholds are round values fixed before adoption, none tuned):
  0-1 RISK_ON, 2-3 NEUTRAL, 4 DETERIORATING, 5-6 RISK_OFF, 7 CRISIS.
  STRONG_RISK_ON is not produced by this rule.  Any missing or non-finite input
  makes the row UNKNOWN (never a partial vote).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from core.nervous_system.contracts.enums import MarketRegime

VOTE_VERSION = "market-regime-vote@1"
VOTE_COUNT = 7

UNCLASSIFIED_REASON = "MARKET_REGIME_UNCLASSIFIED_RULE_VECTOR"


def _flag(value: float) -> bool:
    return value >= 0.5


# name -> (input column, bearish test). Order is the published order.
VOTE_RULES: tuple[tuple[str, str, object], ...] = (
    ("qqq_fast_down", "qqq_fast_down", _flag),
    ("qqq_below_sma50", "qqq_below_sma50", _flag),
    ("spy_trend", "spy_trend_state", lambda v: v <= -0.5),
    ("breadth", "breadth_z", lambda v: v < 0.0),
    ("appetite", "risk_appetite_z", lambda v: v < 0.0),
    ("credit", "credit_risk_z", lambda v: v < 0.0),
    ("vol", "spy_rv20_z", lambda v: v > 1.0),
)
VOTE_INPUT_COLUMNS: tuple[str, ...] = tuple(col for _, col, _ in VOTE_RULES)


@dataclass(frozen=True)
class RegimeVote:
    regime: MarketRegime
    votes: int | None
    bearish: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def reason_codes(self) -> tuple[str, ...]:
        if self.votes is None:
            return (UNCLASSIFIED_REASON,) + tuple(
                f"MARKET_REGIME_VOTE_MISSING_{name.upper()}" for name in self.missing
            )
        return (f"MARKET_REGIME_VOTE_{self.votes}_OF_{VOTE_COUNT}",) + tuple(
            f"MARKET_REGIME_VOTE_BEARISH_{name.upper()}" for name in self.bearish
        )

    def metrics(self) -> dict[str, float]:
        """Vote outputs published alongside the producer metrics (empty when UNKNOWN)."""
        if self.votes is None:
            return {}
        out = {"regime_vote_count": float(self.votes)}
        for name, _, _ in VOTE_RULES:
            out[f"regime_vote_{name}"] = 1.0 if name in self.bearish else 0.0
        return out


def regime_for_votes(votes: int) -> MarketRegime:
    if not 0 <= votes <= VOTE_COUNT:
        raise ValueError(f"vote count {votes} outside 0..{VOTE_COUNT}")
    if votes <= 1:
        return MarketRegime.RISK_ON
    if votes <= 3:
        return MarketRegime.NEUTRAL
    if votes == 4:
        return MarketRegime.DETERIORATING
    if votes <= 6:
        return MarketRegime.RISK_OFF
    return MarketRegime.CRISIS


def _finite_or_none(value: object) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def classify_row(row: Mapping[str, object]) -> RegimeVote:
    """Classify one daily-regime row. Missing/non-finite input -> UNKNOWN."""
    bearish, missing = [], []
    for name, col, is_bearish in VOTE_RULES:
        value = _finite_or_none(row.get(col))
        if value is None:
            missing.append(name)
        elif is_bearish(value):  # type: ignore[operator]
            bearish.append(name)
    if missing:
        return RegimeVote(MarketRegime.UNKNOWN, None, (), tuple(missing))
    return RegimeVote(regime_for_votes(len(bearish)), len(bearish), tuple(bearish), ())


def classify_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorised ``classify_row`` for research and replay: one row per input row,
    columns ``regime_vote_count`` (NaN when any input is missing) and ``regime``."""
    missing_cols = [c for c in VOTE_INPUT_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise KeyError(f"daily-regime frame lacks vote inputs: {missing_cols}")
    rows = [classify_row(r) for r in frame[list(VOTE_INPUT_COLUMNS)].to_dict("records")]
    return pd.DataFrame(
        {
            "regime_vote_count": [np.nan if r.votes is None else float(r.votes) for r in rows],
            "regime": [r.regime.value for r in rows],
        },
        index=frame.index,
    )


__all__ = [
    "RegimeVote",
    "UNCLASSIFIED_REASON",
    "VOTE_COUNT",
    "VOTE_INPUT_COLUMNS",
    "VOTE_RULES",
    "VOTE_VERSION",
    "classify_frame",
    "classify_row",
    "regime_for_votes",
]
