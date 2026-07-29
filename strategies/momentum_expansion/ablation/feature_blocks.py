"""Ablation-arm feature-block definitions (WS-E deliverable 1).

Five arms per the plan (§4 WS-E): baseline / +risk / +liquidity / +sector /
+all. Each additive block is built from
``strategies.momentum_expansion.features.feature_matrix_4h.REGIME_FEATURE_COLUMNS_4H``
(never a separately hand-typed column list) so the arms stay in sync with
whatever WS-B actually exports; a test asserts that invariant.

Grouping used to pick which REGIME_FEATURE_COLUMNS_4H entries belong to
+risk / +liquidity / +sector, cross-checked against the actual join logic in
feature_matrix_4h.py (not just the plan prose — see the taxonomy note in
screen.py for a correction to the plan's own grouping of
sector_above_20d/50d, renamed from the misleading sector_breadth_20/50 --
they are one sector's own above/below-SMA state, not breadth):

  +risk      -> risk_appetite_z + its interaction with ret_20, plus
                market_breadth_z (daily_regime's genuine cross-sector
                breadth_z -- a market-wide regime read with no dedicated arm
                of its own; grouped here rather than left unreachable by any
                arm but +all)
  +liquidity -> liquidity_stress_z, credit_risk_z + liquidity's interaction
                with breakout_20
  +sector    -> sector_excess_21d/63d, sector_rank_21d/63d, sector_rs_accel,
                sector_above_20d/50d, sector_dispersion_21d + the
                rs_spy_20 x sector_rank_63d interaction
  +all       -> REGIME_FEATURE_COLUMNS_4H verbatim (every new column)

``regime_available``/``regime_stale_days`` (the join-health companions, plan
§3) ride along with every non-baseline arm, since a block's own columns are
only interpretable alongside their own availability/staleness context.
"""
from __future__ import annotations

from strategies.momentum_expansion.features.feature_matrix_4h import (
    FEATURE_COLUMNS_4H,
    REGIME_FEATURE_COLUMNS_4H,
)

BASELINE_COLUMNS: list[str] = list(FEATURE_COLUMNS_4H)

_AVAILABILITY_COLUMNS: list[str] = ["regime_available", "regime_stale_days"]

RISK_ADD: list[str] = [
    "risk_appetite_z",
    "int_ret_20_x_risk_appetite_z",
    "market_breadth_z",
    *_AVAILABILITY_COLUMNS,
]

LIQUIDITY_ADD: list[str] = [
    "liquidity_stress_z",
    "credit_risk_z",
    "int_breakout_20_x_liquidity_stress_z",
    *_AVAILABILITY_COLUMNS,
]

SECTOR_ADD: list[str] = [
    "sector_excess_21d",
    "sector_excess_63d",
    "sector_rank_21d",
    "sector_rank_63d",
    "sector_rs_accel",
    "sector_above_20d",
    "sector_above_50d",
    "sector_dispersion_21d",
    "int_rs_spy_20_x_sector_rank_63d",
    *_AVAILABILITY_COLUMNS,
]

ALL_ADD: list[str] = list(REGIME_FEATURE_COLUMNS_4H)


def _dedup_ordered(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


ARMS: dict[str, list[str]] = {
    "baseline":     _dedup_ordered(BASELINE_COLUMNS),
    "+risk":        _dedup_ordered(BASELINE_COLUMNS + RISK_ADD),
    "+liquidity":   _dedup_ordered(BASELINE_COLUMNS + LIQUIDITY_ADD),
    "+sector":      _dedup_ordered(BASELINE_COLUMNS + SECTOR_ADD),
    "+all":         _dedup_ordered(BASELINE_COLUMNS + ALL_ADD),
}

# What each non-baseline arm ADDS relative to baseline (used by screen.py to
# decide which group-appropriate analysis applies to which new column, and by
# tests to verify ARMS == baseline + added with no accidental drops).
ARM_ADDED_COLUMNS: dict[str, list[str]] = {
    "baseline":   [],
    "+risk":      _dedup_ordered(RISK_ADD),
    "+liquidity": _dedup_ordered(LIQUIDITY_ADD),
    "+sector":    _dedup_ordered(SECTOR_ADD),
    "+all":       _dedup_ordered(ALL_ADD),
}


def get_arm_columns(name: str) -> list[str]:
    if name not in ARMS:
        raise KeyError(f"Unknown ablation arm {name!r}; choices: {sorted(ARMS)}")
    return list(ARMS[name])
