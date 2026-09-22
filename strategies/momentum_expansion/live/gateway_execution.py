"""Momentum's governed execution path.

Momentum reuses Meta's router, intent machinery and plan-routing helper rather
than growing a parallel copy. The only genuinely module-specific pieces are:

* the score vocabulary (``mom_score``, not ``s_combo``),
* the snapshot profile id, so momentum's snapshots stay attributable,
* the intent's holding-period and reason codes.

The router refuses PRODUCTION_LIVE in its constructor, so no live account can
be reached through here regardless of configuration.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from core.nervous_system.contracts.enums import InstrumentFamily
from signals.meta_context.meta_ranker.gateway_execution import (
    GovernedPathUnavailable,
    build_router,
)
from signals.meta_context.meta_ranker.nervous_system_adapter import MetaIntentConfig


STRATEGY_ID = "momentum_expansion"
AUDIT_MODULE = "momentum_expansion"

# Momentum's ranking statistic, as `top_n_at` emits it: (ticker,
# expansion_score, rank). `rank` is carried alongside as context; both are
# recorded on the intent as score_components.
#
# NAME COLLISION, deliberate and worth knowing: `expansion_score` is also the
# name of a FORWARD LABEL column in momentum's training matrix (the one that,
# with expansion_target and trend_persistence, gave the parabolic filter a fake
# AUC of 0.951). Here it is the ranker's OUTPUT at the decision bar, computed
# from features only. Same name, opposite direction in time -- do not wire this
# constant into anything that reads the training matrix.
PRIMARY_SCORE = "expansion_score"
SCORE_FIELDS = ("expansion_score", "rank")

# 30 x 4H is momentum's `max_holding_4h_bars` (~6 trading weeks).
EXPECTED_HOLDING_PERIOD = "30x4h"

DEFAULT_PROFILE = "momentum_4h_1420@1"
LATE_SESSION_PROFILE = "momentum_4h_1620@1"


def momentum_intent_config(
    *,
    requested_notional: float | Decimal = Decimal("5000"),
    held_tickers: frozenset[str] = frozenset(),
    instrument_preferences: tuple[InstrumentFamily, ...] | None = None,
) -> MetaIntentConfig:
    """Versioned metadata for momentum entry and reduction intents."""

    return MetaIntentConfig(
        strategy_id=STRATEGY_ID,
        held_tickers=held_tickers,
        requested_notional=Decimal(str(requested_notional)),
        model_version="momentum-expansion-model@1",
        feature_version="momentum-features-4h@1",
        config_version="momentum-expansion-intent@1",
        instrument_preferences=(
            instrument_preferences
            if instrument_preferences is not None
            else (InstrumentFamily.EQUITY, InstrumentFamily.SINGLE_OPTION)
        ),
        expected_holding_period=EXPECTED_HOLDING_PERIOD,
        reason_codes=("MOMENTUM_TOP_N", "MOMENTUM_LONG_ENTRY"),
        primary_score=PRIMARY_SCORE,
        score_fields=SCORE_FIELDS,
    )


def build_momentum_router(
    *,
    requested_notional: float | Decimal = Decimal("5000"),
    held_tickers: frozenset[str] = frozenset(),
    profile_id: str = DEFAULT_PROFILE,
    environ: Mapping[str, str] | None = None,
    clock: Any = None,
):
    """Assemble momentum's governed path, or raise GovernedPathUnavailable."""

    return build_router(
        intent_config=momentum_intent_config(
            requested_notional=requested_notional, held_tickers=held_tickers
        ),
        environ=environ,
        clock=clock,
        required_snapshot_profile=profile_id,
    )


def momentum_scores_by_ticker(scored: Any) -> dict[str, dict[str, float]]:
    """Map momentum's scored frame to the score vocabulary intents record.

    An entry whose ``mom_score`` is missing is deliberately left out: the
    adapter refuses to open a position it cannot explain, and a fabricated
    score would defeat that.
    """

    out: dict[str, dict[str, float]] = {}
    if scored is None or getattr(scored, "empty", True):
        return out
    frame = scored.reset_index() if "ticker" not in scored.columns else scored
    for _index, row in frame.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        components: dict[str, float] = {}
        for name in SCORE_FIELDS:
            value = row.get(name)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number != number:  # NaN
                continue
            components[name] = number
        if PRIMARY_SCORE in components:
            out[ticker] = components
    return out


__all__ = [
    "AUDIT_MODULE",
    "DEFAULT_PROFILE",
    "GovernedPathUnavailable",
    "LATE_SESSION_PROFILE",
    "PRIMARY_SCORE",
    "SCORE_FIELDS",
    "STRATEGY_ID",
    "build_momentum_router",
    "momentum_intent_config",
    "momentum_scores_by_ticker",
]
