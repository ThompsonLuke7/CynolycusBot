"""Decide whether a price source may be used to compute option P&L.

Pure: no clock, no IO, no network. The verdict is a function of the measured
metrics and the thresholds they were judged against, so it reproduces exactly
from a stored report.

The gate runs *before* any P&L, not after. That ordering is the whole point —
the retracted 2026-07 study computed confident numbers first and discovered the
source could not support them later.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.nervous_system.contracts.replay import (
    FABRICATED_MARKS,
    FIT_MARKS,
    TRADE_DERIVED_MARKS,
    FitnessReason,
    SideFitnessMetrics,
    SourceFitnessReport,
    SourceFitnessStatus,
    SourceFitnessThresholds,
)


REQUIRED_SIDES = ("CALL", "PUT")

# Reasons that mean "this source is broken", as opposed to "we cannot tell
# yet". A defect outranks a thin sample: reporting a broken source as merely
# unproven would invite somebody to go and gather more of the same bad data.
_DEFECT_REASONS = frozenset(
    {
        FitnessReason.TRADE_ONLY,
        FitnessReason.SYNTHETIC_MARKS,
        FitnessReason.ENTITLEMENT_UNVERIFIED,
        FitnessReason.STALE_MARKS,
        FitnessReason.CROSSED_QUOTES,
        FitnessReason.LOW_DERIVATIVE_CORRELATION,
        FitnessReason.SIDE_NOT_EVALUATED,
    }
)

_SAMPLE_REASONS = frozenset(
    {FitnessReason.INSUFFICIENT_POSITIONS, FitnessReason.INSUFFICIENT_SESSIONS}
)


def _side_reasons(
    metrics: SideFitnessMetrics, thresholds: SourceFitnessThresholds
) -> tuple[set[FitnessReason], set[FitnessReason]]:
    reasons: set[FitnessReason] = set()
    warnings: set[FitnessReason] = set()

    if metrics.mark_type in TRADE_DERIVED_MARKS:
        # A trade-bar feed only has bars on days something traded; the "price"
        # at an arbitrary timestamp is a stale last print, not a mark.
        reasons.add(FitnessReason.TRADE_ONLY)
    elif metrics.mark_type in FABRICATED_MARKS:
        reasons.add(FitnessReason.SYNTHETIC_MARKS)
    elif metrics.mark_type not in FIT_MARKS:
        # Anything not affirmatively a two-sided quote is refused rather than
        # assumed acceptable, so a new mark type defaults to unfit.
        reasons.add(FitnessReason.SYNTHETIC_MARKS)

    if not metrics.entitlement_verified:
        # A 200 response is not evidence of fitness for purpose.
        reasons.add(FitnessReason.ENTITLEMENT_UNVERIFIED)
    if metrics.valid_quote_fraction < thresholds.min_valid_quote_fraction:
        reasons.add(FitnessReason.CROSSED_QUOTES)
    if metrics.identical_mark_fraction > thresholds.max_identical_mark_fraction:
        # Entry and exit marks that are the same number are the signature of a
        # series that is not tracking value.
        reasons.add(FitnessReason.STALE_MARKS)
    if metrics.max_quote_age_seconds > thresholds.max_quote_age_seconds:
        reasons.add(FitnessReason.STALE_MARKS)
    if metrics.pearson < thresholds.min_pearson:
        reasons.add(FitnessReason.LOW_DERIVATIVE_CORRELATION)
    elif metrics.pearson < thresholds.warn_pearson:
        warnings.add(FitnessReason.CORRELATION_BELOW_WARNING_BAND)

    if metrics.matched_positions < thresholds.min_matched_positions:
        reasons.add(FitnessReason.INSUFFICIENT_POSITIONS)
    if metrics.sessions < thresholds.min_sessions:
        reasons.add(FitnessReason.INSUFFICIENT_SESSIONS)

    return reasons, warnings


def evaluate_source_fitness(
    *,
    sides: Sequence[SideFitnessMetrics],
    thresholds: SourceFitnessThresholds,
    source: str,
    feed: str,
    tier: str,
) -> SourceFitnessReport:
    """Return the fitness verdict for one option price source.

    Thresholds are enforced per side, never in aggregate: averaging a broken
    put series against a healthy call series hides the break.
    """

    if not isinstance(thresholds, SourceFitnessThresholds):
        raise TypeError("thresholds must be a SourceFitnessThresholds")

    reasons: set[FitnessReason] = set()
    warnings: set[FitnessReason] = set()

    seen = {metrics.option_type for metrics in sides}
    for required in REQUIRED_SIDES:
        if required not in seen:
            # An unmeasured side is not a passing side.
            reasons.add(FitnessReason.SIDE_NOT_EVALUATED)

    for metrics in sides:
        side_reasons, side_warnings = _side_reasons(metrics, thresholds)
        reasons |= side_reasons
        warnings |= side_warnings

    if reasons & _DEFECT_REASONS:
        status = SourceFitnessStatus.SOURCE_UNFIT_FOR_OPTION_PNL
    elif reasons & _SAMPLE_REASONS:
        status = SourceFitnessStatus.SOURCE_FITNESS_INSUFFICIENT_SAMPLE
    else:
        status = SourceFitnessStatus.FIT_FOR_OPTION_PNL

    # Sorted and deduplicated: the report is content-hashed downstream, so the
    # same failure must always render identically.
    return SourceFitnessReport(
        status=status,
        reasons=tuple(sorted(reasons, key=lambda reason: reason.value)),
        warnings=tuple(sorted(warnings, key=lambda reason: reason.value)),
        sides=tuple(sides),
        thresholds_hash=thresholds.content_hash(),
        source=source,
        feed=feed,
        tier=tier,
    )


__all__ = ["REQUIRED_SIDES", "evaluate_source_fitness"]
